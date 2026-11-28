"""Encoder mới (Densenet128_FER+) cho web. Visual: crop mặt người nói bằng chuyển động môi."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SR = 16000


def pick_frames(items, n):
    if n <= 0 or len(items) <= n:
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


class SpeakerFaceCropper:
    """Chọn mặt người nói = mặt có độ mở miệng thay đổi nhiều nhất trong [start, end].
    Crop VUÔNG (viền ~15%). Tạo một lần cho mỗi lượt predict, nhớ gọi close()."""

    def __init__(self):
        import mediapipe as mp
        self.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=6, min_detection_confidence=0.5)

    def close(self):
        self.fm.close()

    def __call__(self, cap, start, end, n_sample=24, n_out=8):
        import cv2
        tracks = []
        for k, t in enumerate(np.linspace(start, max(start, end - 0.05), n_sample)):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            for lm in self.fm.process(rgb).multi_face_landmarks or []:
                pts = np.array([(p.x * W, p.y * H) for p in lm.landmark])
                (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                h = max(y1 - y0, 1.0)
                size = max(x1 - x0, h)
                opening = abs(pts[14, 1] - pts[13, 1]) / h
                c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
                half = size * 0.575
                crop = rgb[int(max(c[1] - half, 0)):int(min(c[1] + half, H)),
                           int(max(c[0] - half, 0)):int(min(c[0] + half, W))]
                if crop.size == 0:
                    continue
                best, bd = None, 0.6 * size
                for tr in tracks:
                    d = np.linalg.norm(tr['c'] - c)
                    if k not in tr['frames'] and d < bd:
                        best, bd = tr, d
                if best is None:
                    best = {'c': c, 'frames': set(), 'open': [], 'size': [], 'img': []}
                    tracks.append(best)
                best['c'] = c
                best['frames'].add(k)
                best['open'].append(opening)
                best['size'].append(size)
                best['img'].append(Image.fromarray(crop))
        if not tracks:
            return []
        moving = [t for t in tracks if len(t['open']) >= 3]
        t = max(moving, key=lambda t: np.std(t['open'])) if moving \
            else max(tracks, key=lambda t: np.mean(t['size']))
        return pick_frames(t['img'], n_out)


class OnlineEncodersV2:
    def __init__(self, args, device):
        d = str(Path(args.encoder_dir).resolve())
        if d not in sys.path:
            sys.path.insert(0, d)
        from encoders import AudioEncoder, FaceEncoder, OpenSmileAudioEncoder, TextEncoder
        from transformers import AutoTokenizer

        fd = Path(args.feat_dir)
        self.device = device
        self.audio_mode = getattr(args, "audio_mode", "auto")
        # Accept flat MELD artifacts and organized IEMOCAP artifacts.
        norm_path = fd / "norm.npz"
        if not norm_path.exists() and (fd / "stats" / "norm.npz").exists():
            norm_path = fd / "stats" / "norm.npz"
        speaker_path = fd / "speaker_map.json"
        if not speaker_path.exists() and (fd / "splits" / "speaker_map.json").exists():
            speaker_path = fd / "splits" / "speaker_map.json"
        audio_head_path = fd / "audio_head.npz"
        has_meld_audio = audio_head_path.exists()
        has_iemocap_audio = norm_path.exists() and {"a_mean", "a_std"}.issubset(set(np.load(norm_path).files))
        if self.audio_mode == "auto":
            self.audio_mode = "meld" if has_meld_audio else "iemocap" if has_iemocap_audio else "meld"
        if self.audio_mode not in ("meld", "iemocap"):
            raise ValueError("audio_mode must be 'auto', 'meld' or 'iemocap'")

        print('[load] RoBERTa-base (text_ft)...', flush=True)
        self.text_model = TextEncoder.load(str(args.text_ckpt), device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model.base)

        if self.audio_mode == "meld":
            print('[load] Wav2Vec2-base + audio_head (MELD)...', flush=True)
            self.audio_mod = AudioEncoder(args.audio_model).to(device)
            self.audio_mod.load_head(str(audio_head_path))
            self.audio_dim = 300
        else:
            print('[load] openSMILE IS10 (IEMOCAP)...', flush=True)
            n_audio = np.load(norm_path)
            self.audio_mod = OpenSmileAudioEncoder(n_audio['a_mean'], n_audio['a_std'])
            self.audio_dim = 1582

        print('[load] DenseNet FER+...', flush=True)
        self.face = FaceEncoder(str(args.face_ckpt), device)
        n = np.load(norm_path)
        if 'v_mean' not in n or 'v_std' not in n:
            raise KeyError(f"Missing v_mean/v_std in {norm_path}")
        self.v_mean, self.v_std = n['v_mean'], n['v_std']

        raw = json.load(open(speaker_path, encoding='utf-8'))
        self.other_slot = raw['__other__']
        self.speaker_map = {k.strip().lower(): v for k, v in raw.items() if k != '__other__'}
        self.max_seconds = 15
        print('[load] Encoders V2 ready', flush=True)

    @torch.inference_mode()
    def text(self, s: str) -> np.ndarray:
        tok = self.tokenizer([s], padding=True, truncation=True, max_length=128,
                             return_tensors='pt').to(self.device)
        out = self.text_model.features(tok['input_ids'], tok['attention_mask'])[0]
        return out.float().cpu().numpy().astype(np.float32)

    def _prep(self, wav):
        w = torch.from_numpy(np.array(wav[:SR * self.max_seconds], dtype=np.float32))
        if w.numel() < SR // 10:
            w = F.pad(w, (0, SR // 10 - w.numel()))
        return (w - w.mean()) / torch.sqrt(w.var() + 1e-7)   # giống encoders.load_wave

    @torch.inference_mode()
    def audio(self, wav) -> np.ndarray:
        if self.audio_mode == "iemocap":
            raw = self.audio_mod.extract_signal(np.asarray(wav, dtype=np.float32), SR)
            return self.audio_mod.normalize(raw)
        x = self._prep(wav)[None].to(self.device)
        lens = torch.tensor([x.shape[1]], device=self.device)
        raw = self.audio_mod._pool(x, lens).float().cpu().numpy()[0]
        return self.audio_mod.project(raw)

    @torch.inference_mode()
    def w2v_hidden(self, wav):
        if self.audio_mode != "meld":
            raise RuntimeError("w2v_hidden is unavailable in IEMOCAP openSMILE mode")
        x = self._prep(wav)[None].to(self.device)
        return self.audio_mod.w2v(x, output_hidden_states=True).hidden_states

    @torch.inference_mode()
    def visual(self, crops) -> np.ndarray:
        if not crops:
            return np.zeros(342, np.float32)   # không thấy mặt = trung bình train sau chuẩn hóa
        emb = self.face.embed_pil(crops).mean(0)
        return ((emb - self.v_mean) / self.v_std).astype(np.float32)