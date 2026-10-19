"""Windows PC turn-based MELD / DiGemo live demo: microphone + camera -> multimodal emotion.

Uses the exact RoBERTa / Wav2Vec2 / DenseNet feature definitions from erc_pipeline.py.
Requires online_{audio,visual}_transform.joblib matching the existing .npy features.
Only the current user is observed (speaker slot 8 = MELD 'Other' by default).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from predict import load_model_from_checkpoint
from recover_feature_transforms import transform_one

EMOTIONS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']


def choose_torch_device(name: str) -> str:
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; set --torch-device cpu')
    return name


def safe_weights(path: Path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class OnlineEncoders:
    def __init__(self, args, device):
        import cv2
        import joblib
        from PIL import Image
        from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoFeatureExtractor
        from torchvision import models
        from torchvision.models import DenseNet121_Weights

        self.cv2 = cv2
        self.Image = Image
        self.device = device
        self.audio_transform = joblib.load(args.features_dir / 'online_audio_transform.joblib')
        self.visual_transform = joblib.load(args.features_dir / 'online_visual_transform.joblib')
        if self.audio_transform['output_dim'] != 300 or self.visual_transform['output_dim'] != 342:
            raise ValueError('MELD DiGemo expects audio=300 and visual=342')

        print('[load] RoBERTa-large tokenizer + fine-tuned encoder...', flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        conf = AutoConfig.from_pretrained(args.text_model)
        self.text_enc = AutoModel.from_config(conf, add_pooling_layer=False)
        self.text_head = torch.nn.Linear(conf.hidden_size, len(EMOTIONS))
        state = safe_weights(args.roberta_checkpoint)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # Check full fine-tuned classifier from erc_pipeline.Clf with enc.*, drop.*, head.*
        enc_state = {k[4:]: v for k, v in state.items() if k.startswith('enc.')}
        if not enc_state or not any(k.startswith('head.') for k in state):
            raise ValueError('Expected roberta_meld_ft.pt from erc_pipeline.py (enc.*, head.*)')
        self.text_enc.load_state_dict(enc_state, strict=True)
        self.text_head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith('head.')}, strict=True)
        self.text_enc = self.text_enc.eval().to(device)
        if conf.hidden_size != 1024:
            raise ValueError('Text encoder is not 1024-dimension roberta-large')

        print('[load] Wav2Vec2 model...', flush=True)
        self.audio_pre = AutoFeatureExtractor.from_pretrained(args.audio_model)
        if self.audio_pre.sampling_rate != 16000:
            raise ValueError(f'Expected 16000 Hz audio model, got {self.audio_pre.sampling_rate}')
        self.audio_enc = AutoModel.from_pretrained(args.audio_model).eval().to(device)

        print('[load] DenseNet121 model...', flush=True)
        weights = DenseNet121_Weights.IMAGENET1K_V1
        self.visual_enc = models.densenet121(weights=weights)
        self.visual_enc.classifier = torch.nn.Identity()
        self.visual_enc = self.visual_enc.eval().to(device)
        self.visual_pre = weights.transforms()
        self.n_frames = args.num_frames
        self.max_seconds = 15

        print('[load] All encoders ready', flush=True)

    @torch.inference_mode()
    def text(self, sentence: str) -> np.ndarray:
        tok = self.tokenizer(sentence, padding=True, truncation=True, max_length=96, return_tensors='pt')
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.text_enc(**tok).last_hidden_state[:, 0]
        return out[0].float().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def audio(self, waveform_16k: np.ndarray) -> np.ndarray:
        wav = np.asarray(waveform_16k[:16000 * self.max_seconds], dtype=np.float32)
        inp = self.audio_pre(wav, sampling_rate=16000, return_tensors='pt')
        inp = {k: v.to(self.device) for k, v in inp.items()}
        hidden = self.audio_enc(**inp, output_hidden_states=True).hidden_states[-1]
        raw = hidden.squeeze(0).mean(dim=0).float().cpu().numpy().astype(np.float32)
        return transform_one(self.audio_transform, raw)

    @torch.inference_mode()
    def visual(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        idx = np.linspace(0, len(frames_bgr)-1, num=self.n_frames, dtype=int)
        inputs = torch.stack([
            self.visual_pre(self.Image.fromarray(self.cv2.cvtColor(frames_bgr[int(i)], self.cv2.COLOR_BGR2RGB)))
            for i in idx
        ]).to(self.device)
        raw = self.visual_enc(inputs).mean(0).float().cpu().numpy().astype(np.float32)
        return transform_one(self.visual_transform, raw)


def resample_to_16k(wav, src_sr):
    from scipy.signal import resample_poly
    src_sr = int(round(src_sr))
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if src_sr == 16000:
        return wav
    g = math.gcd(16000, src_sr)
    return np.asarray(resample_poly(wav, 16000 // g, src_sr // g), dtype=np.float32)


def open_camera(idx):
    import cv2
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open webcam index {idx}; try --camera-index 1 or check Windows camera permissions')
    return cap


def record_turn(cap, microphone, max_seconds: float):
    """Turn-based capture with audio callback plus sampling webcam; SPACE finishes early."""
    import cv2
    import sounddevice as sd

    mic_info = sd.query_devices(microphone, 'input')
    samplerate = int(round(mic_info['default_samplerate']))
    if samplerate <= 0:
        raise RuntimeError('Invalid microphone sample rate')
    chunks = []
    frames = []
    last = -1.0

    def callback(indata, nframes, tinfo, status):
        chunks.append(indata[:, 0].copy())

    print(f'[record] Mic: {mic_info["name"]}, {samplerate} Hz | SPACE to stop early', flush=True)
    with sd.InputStream(device=microphone, channels=1, samplerate=samplerate,
                        dtype='float32', callback=callback, blocksize=0):
        start = time.perf_counter()
        while time.perf_counter() - start < max_seconds:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Webcam stopped returning frames')
            elapsed = time.perf_counter() - start
            if elapsed - last >= 0.12:
                frames.append(frame.copy())
                last = elapsed
            display = frame.copy()
            cv2.putText(display, f'RECORDING {elapsed:.1f}/{max_seconds:.0f}s | SPACE: stop',
                        (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.imshow('DiGemo webcam', display)
            if cv2.waitKey(1) & 0xFF == 32:
                break

    if not chunks or not frames:
        raise RuntimeError('No audio or video captured')
    wav = resample_to_16k(np.concatenate(chunks), samplerate)
    duration = len(wav) / 16000
    rms = float(np.sqrt(np.mean(wav ** 2)))
    print(f'[record] {duration:.2f}s audio, {len(frames)} candidate frames, RMS={rms:.5f}', flush=True)
    if duration < 0.35 or rms < 0.001:
        raise ValueError('Audio too short or almost silent; check microphone and try again')
    return wav, frames


@torch.inference_mode()
def model_predict(model, features, speaker_index, n_speakers, device):
    n = len(features)
    if n == 0:
        raise ValueError('Empty utterance history')
    def stacked(col):
        return torch.from_numpy(np.stack([f[col] for f in features])).unsqueeze(1).to(device)
    qmask = torch.zeros((n, 1, n_speakers), dtype=torch.float32, device=device)
    qmask[:, 0, speaker_index] = 1.
    umask = torch.ones((n, 1), dtype=torch.float32, device=device)
    logits, _, _, _, _ = model(stacked('text'), stacked('visual'), stacked('audio'),
                              umask, qmask, [n])
    prob = torch.softmax(logits[-1], -1).float().cpu().numpy()
    return int(prob.argmax()), prob


def main():
    ap = argparse.ArgumentParser(description='Live MELD DiGemo emotion demo')
    ap.add_argument('--checkpoint', default='checkpoints/best_model_MELD_260.pth')
    ap.add_argument('--features-dir', type=Path, default=Path('features'))
    ap.add_argument('--roberta-checkpoint', type=Path, default=None)
    ap.add_argument('--text-model', default='roberta-large')
    ap.add_argument('--audio-model', default='facebook/wav2vec2-base-960h')
    ap.add_argument('--torch-device', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--camera-index', type=int, default=0)
    ap.add_argument('--mic-index', type=int, default=None)
    ap.add_argument('--list-mics', action='store_true')
    ap.add_argument('--seconds', type=float, default=7.0)
    ap.add_argument('--num-frames', type=int, default=8)
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--speaker-index', type=int, default=8,
                    help='MELD: 0..7 are frequent speakers; 8 is Other (default)')
    ap.add_argument('--manual-transcript', action='store_true',
                    help='Type transcript for debugging instead of loading Whisper')
    ap.add_argument('--whisper-model', default='base.en',
                    help='faster-whisper English ASR; use tiny.en for lower CPU latency')
    ap.add_argument('--jsonl', type=Path, default=None,
                    help='Optional: log transcripts and predicted emotions only')
    args = ap.parse_args()

    if args.list_mics:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if not 0 < args.seconds <= 15:
        ap.error('--seconds must be between 0 and 15 (audio encoder max is 15s)')
    if args.window < 1:
        ap.error('--window must be >=1')
    if not 0 <= args.speaker_index < 9:
        ap.error('--speaker-index must be 0..8 for MELD')
    if args.num_frames < 1:
        ap.error('--num-frames must be >=1')

    args.roberta_checkpoint = args.roberta_checkpoint or args.features_dir / 'roberta_meld_ft.pt'
    needed = [Path(args.checkpoint), args.roberta_checkpoint,
              args.features_dir / 'online_audio_transform.joblib',
              args.features_dir / 'online_visual_transform.joblib']
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}. Read PHASE2_SETUP.md before running live demo.')

    device = choose_torch_device(args.torch_device)
    model, cfg, dims, nc, ns, td = load_model_from_checkpoint(args.checkpoint, device)
    if cfg.dataset != 'MELD' or dims != [1024, 342, 300] or nc != 7 or ns != 9:
        raise ValueError('This script supports only MELD checkpoint [1024,342,300], 7 classes, 9 speakers')
    if getattr(cfg, 'modals', 'tva') != 'tva':
        raise ValueError('The current demo requires a 3-modality tva checkpoint')
    print(f'[load] DiGemo checkpoint={args.checkpoint}, torch={device}', flush=True)

    encoder = OnlineEncoders(args, device)
    asr = None
    if not args.manual_transcript:
        from faster_whisper import WhisperModel
        print(f'[load] faster-whisper {args.whisper_model} (CPU int8)...', flush=True)
        asr = WhisperModel(args.whisper_model, device='cpu', compute_type='int8')

    cap = open_camera(args.camera_index)
    history = deque(maxlen=args.window)
    try:
        while True:
            cmd = input('\nENTER: record a turn | r: reset dialogue | q: quit > ').strip().lower()
            if cmd == 'q':
                break
            if cmd == 'r':
                history.clear()
                print('[history] cleared')
                continue
            if cmd:
                continue
            try:
                audio, frames = record_turn(cap, args.mic_index, args.seconds)
                t0 = time.perf_counter()
                if args.manual_transcript:
                    transcript = input('English transcript > ').strip()
                else:
                    segments, _ = asr.transcribe(audio, language='en', beam_size=3,
                                                 vad_filter=False, condition_on_previous_text=False)
                    transcript = ' '.join(seg.text.strip() for seg in segments).strip()
                t_asr = time.perf_counter() - t0
                if not transcript:
                    print('[skip] ASR found no speech; record another turn')
                    continue
                print(f'[ASR] {transcript}', flush=True)
                t1 = time.perf_counter()
                vec_t = encoder.text(transcript)
                vec_a = encoder.audio(audio)
                vec_v = encoder.visual(frames)
                if (vec_t.shape, vec_v.shape, vec_a.shape) != ((1024,), (342,), (300,)):
                    raise ValueError(f'Bad features: text={vec_t.shape}, visual={vec_v.shape}, audio={vec_a.shape}')
                t_encoder = time.perf_counter() - t1
                history.append({'text': vec_t, 'visual': vec_v, 'audio': vec_a})
                t2 = time.perf_counter()
                pred, prob = model_predict(model, history, args.speaker_index, ns, device)
                t_digemo = time.perf_counter() - t2
                top = np.argsort(prob)[::-1][:3]
                print(f'\n[DiGemo] emotion={EMOTIONS[pred]} ({prob[pred]:.1%}), history={len(history)} utterances')
                print('  ' + ', '.join(f'{EMOTIONS[int(j)]}: {prob[j]:.1%}' for j in top))
                print(f'  latency: ASR={t_asr:.2f}s | encoders={t_encoder:.2f}s | DiGemo={t_digemo:.3f}s')
                if args.jsonl:
                    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                    record = {'time': datetime.now().isoformat(), 'text': transcript,
                              'emotion': EMOTIONS[pred], 'probabilities': dict(zip(EMOTIONS, prob.astype(float).tolist())),
                              'history_size': len(history),
                              'latency_s': {'asr': t_asr, 'encoders': t_encoder, 'digemo': t_digemo}}
                    with args.jsonl.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
            except (ValueError, RuntimeError) as exc:
                print(f'[turn failed] {exc} -- try again (history unchanged unless inference failed)')
    finally:
        import cv2
        cap.release()
        cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
