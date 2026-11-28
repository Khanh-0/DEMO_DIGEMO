"""YuNet + IoU + Light-ASD block processor for isolated streaming mode."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import sys
import threading
import cv2
import numpy as np
import torch
from python_speech_features import mfcc
from .track_state import TrackState, ReIDEntry


_LIGHT_ASD_IMPORT_LOCK = threading.Lock()


class ASDStream:
    def __init__(self, root: Path, width=640, height=360, fps=30, context=37, max_gap_frames=4):
        self.root, self.width, self.height, self.fps, self.context = Path(root), width, height, fps, context
        self.max_gap_frames = int(max_gap_frames)
        # The route passes the project root (demo_meld), while this runtime
        # lives under demo_meld/main. Resolve both layouts for old jobs.
        self.main_root = self.root / "main" if (self.root / "main" / "Light-ASD").is_dir() else self.root
        self.asd_root = self.main_root / "Light-ASD"
        self.weight = self.asd_root / "weight" / "pretrain_AVA_CVPR.model"
        self.yunet = self.main_root / "face_detection_yunet_2023mar.onnx"
        if not self.asd_root.is_dir():
            raise RuntimeError(f"Light-ASD folder not found: {self.asd_root}")
        if not self.weight.is_file():
            raise RuntimeError(f"Light-ASD checkpoint not found: {self.weight}")
        if not self.yunet.is_file():
            raise RuntimeError(f"YuNet model not found: {self.yunet}")
        with _LIGHT_ASD_IMPORT_LOCK:
            import importlib
            sys.path.insert(0, str(self.asd_root))
            # Other project components also expose a top-level `model` module.
            # Remove a stale non-package before loading Light-ASD's package.
            stale_model = sys.modules.get("model")
            if stale_model is not None and not hasattr(stale_model, "__path__"):
                sys.modules.pop("model", None)
            ASD_Model = importlib.import_module("model.Model").ASD_Model
            lossAV = importlib.import_module("loss").lossAV
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = ASD_Model().to(self.device), lossAV().to(self.device)
        state = torch.load(self.weight, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        net.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")})
        head.load_state_dict({k[7:]: v for k, v in state.items() if k.startswith("lossAV.")})
        net.eval(); head.eval(); self.net, self.head = net, head
        self.detector = cv2.FaceDetectorYN.create(str(self.yunet), "", (width, height), .65, .3, 5000)
        self.tracks: dict[int, TrackState] = {}
        self.reid_cache: dict[int, ReIDEntry] = {}
        self.next_id = 0
        self.track_expire_frames = 30
        self.reid_ttl_frames = int(8 * fps)
        self.prev_frame = -1

    @staticmethod
    def _iou(a, b):
        x, y = max(a[0], b[0]), max(a[1], b[1]); r, d = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, r-x) * max(0, d-y)
        return inter / max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

    def _display_box(self, box):
        """Expand the detector face box for stable face crops and UI display."""
        x1, y1, x2, y2 = map(float, box)
        w, h = x2 - x1, y2 - y1
        # Keep the detector box for IoU; add stable context around the face.
        return [max(0, x1 - .20*w), max(0, y1 - .15*h), min(self.width, x2 + .20*w), min(self.height, y2 + .35*h)]

    def _crop(self, frame, box):
        # Track boxes are [x1,y1,x2,y2]; crop from the expanded display box.
        x1, y1, x2, y2 = map(int, self._display_box(box))
        if x2 <= x1 or y2 <= y1: return None
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (112, 112))

    @torch.inference_mode()
    def _scores(self, rows, audio):
        n = len(rows); samples_per_frame = 16000 / self.fps
        start = int(rows[0][0] * samples_per_frame); needed = int(n * samples_per_frame)
        pcm = np.pad(audio[start:start+needed], (0, max(0, needed + int(.015 * 16000) - len(audio[start:start+needed]))))
        af = mfcc(pcm, 16000, numcep=13, winlen=.025, winstep=.01)[:4*n]
        af = np.pad(af, ((0, max(0, 4*n-len(af))), (0, 0)))[:4*n]
        a = torch.tensor(af, dtype=torch.float32, device=self.device)[None]
        v = torch.tensor(np.stack([r[2] for r in rows]), dtype=torch.float32, device=self.device)[None]
        out = self.net.forward_audio_visual_backend(self.net.forward_audio_frontend(a), self.net.forward_visual_frontend(v))
        return self.head.FC(out).softmax(-1)[:, 1].cpu().numpy()

    @staticmethod
    def _appearance(crop):
        if crop is None: return None
        small = cv2.resize(crop, (16, 16)).astype(np.float32).reshape(-1)
        norm = np.linalg.norm(small)
        return small / norm if norm else None

    @staticmethod
    def _cosine(a, b):
        if a is None or b is None or len(a) != len(b): return -1.0
        return float(np.dot(a, b) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))

    def _expire(self, frame_index):
        for tid in list(self.tracks):
            state = self.tracks[tid]
            state.missing_frames = max(0, frame_index - state.last_seen_frame)
            if state.missing_frames > self.track_expire_frames:
                if state.embedding_centroid is not None:
                    self.reid_cache[tid] = ReIDEntry(tid, state.embedding_centroid.copy(), frame_index + self.reid_ttl_frames)
                del self.tracks[tid]
        for tid in list(self.reid_cache):
            if self.reid_cache[tid].expires_frame < frame_index:
                del self.reid_cache[tid]

    def crops_for_track(self, track_id: int, end_frame: int, seconds: float = 1.0, count: int = 8):
        track = self.tracks.get(int(track_id))
        if not track: return []
        lo = end_frame - int(seconds * self.fps)
        rows = [row for row in track["rows"] if lo <= row[0] <= end_frame]
        if not rows: return []
        indices = np.unique(np.linspace(0, len(rows)-1, min(count, len(rows))).round().astype(int))
        return [rows[i][2] for i in indices]

    def process_block(self, cap, audio, start_frame, end_frame):
        boxes = []
        for frame_index in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index); ok, frame = cap.read()
            if not ok: continue
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            _, dets = self.detector.detect(frame)
            current = []
            for d in dets if dets is not None else []:
                x, y, w, h = map(float, d[:4]); current.append([x, y, x+w, y+h])
            matches = sorted((self._iou(t["box"], b), tid, j) for tid,t in self.tracks.items() if frame_index - t.last_seen_frame <= self.max_gap_frames for j,b in enumerate(current))[::-1]
            assigned, used_t, used_b = {}, set(), set()
            for overlap, tid, j in matches:
                if overlap < .25: break
                if tid not in used_t and j not in used_b: assigned[j] = tid; used_t.add(tid); used_b.add(j)
            for j, box in enumerate(current):
                tid = assigned.get(j)
                crop = self._crop(frame, box)
                appearance = self._appearance(crop)
                if tid is None:
                    candidates = [(self._cosine(appearance, entry.embedding_centroid), old_tid) for old_tid, entry in self.reid_cache.items()]
                    best_score, best_tid = max(candidates, default=(-1.0, None))
                    if best_tid is not None and best_score >= .72:
                        tid = best_tid; del self.reid_cache[tid]
                    else:
                        tid = self.next_id; self.next_id += 1
                    self.tracks[tid] = TrackState(tid, box, frame_index, rows=deque(maxlen=self.context))
                track = self.tracks[tid]
                track.update(box=box, last_seen_frame=frame_index, missing_frames=0)
                if crop is not None:
                    track.rows.append((frame_index, box, crop))
                    track.box_history.append((frame_index, list(box)))
                    track.recent_crops.append(crop)
                    if appearance is not None: track.update_embedding(appearance)

        self._expire(end_frame)
        scored = []
        for tid, track in self.tracks.items():
            if track.last_seen_frame >= start_frame:
                rows = list(track["rows"])
                value = float(np.mean(self._scores(rows, audio))) if len(rows) >= 5 else 0.0
                track.asd_score = value
                last_box = track.get("box", [0, 0, 0, 0])
                display_box = self._display_box(last_box)
                track.display_box = display_box
                scored.append({"track_id": tid, "box": display_box, "tracking_box": last_box, "asd_score": value, "speaking": value >= .35, "start": start_frame/self.fps, "end": end_frame/self.fps})
        scored_ids = {item["track_id"] for item in scored}
        for tid, track in self.tracks.items():
            if tid not in scored_ids and track.last_seen_frame >= start_frame:
                display_box = self._display_box(track.get("box", [0,0,0,0]))
                scored.append({"track_id": tid, "box": display_box, "tracking_box": track.get("box", [0,0,0,0]), "asd_score": 0.0, "speaking": False, "start": start_frame/self.fps, "end": end_frame/self.fps})
        return scored
