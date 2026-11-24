"""Suy luận: thư mục video mp4 (diaX_uttY.mp4) + csv -> dự đoán cảm xúc từng câu bằng DiGEMO đã train.

Đặt file này cạnh model.py / module.py / utils.py của DiGEMO. Dùng đúng encoder + scaler đã dùng lúc build pkl:
  text  : text_ft/best.pt          audio: feats/audio_head.npz
  visual: ckpt_fer/best.pt + feats/norm.npz     speaker: feats/speaker_map.json

csv cần cột Dialogue_ID, Utterance_ID, Speaker; cột Utterance nếu thiếu/rỗng thì dùng faster-whisper (pip install faster-whisper).
Visual: MTCNN không cần, dùng mediapipe FaceMesh (pip install mediapipe opencv-python); chọn người nói = mặt có
độ mở miệng thay đổi nhiều nhất theo thời gian (chuyển động môi); không thấy mặt nào -> feature visual = trung bình train.

    python infer.py --csv test_sent_emo.csv --video_dir <thư mục mp4 test> --digemo digemo_best.pt --out pred.csv ^
        --hidden_dim 512 ...   (các siêu tham số DiGEMO PHẢI trùng lúc train)
"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from common import EMO_NAMES, collect_files, key
from encoders import AudioEncoder, FaceEncoder, TextEncoder, pick_frames, text_features


# ----------------------------------------------------------------------------- chọn mặt người nói bằng chuyển động môi
def speaker_crops(video, n_sample=24, n_out=8):
    import cv2
    import mediapipe as mp
    from PIL import Image

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.unique(np.linspace(0, max(total - 1, 0), n_sample).astype(int))
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=6, min_detection_confidence=0.5)
    tracks = []  # mỗi track: {"c": tâm, "frames": set, "open": [], "size": [], "img": []}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        for lm in fm.process(rgb).multi_face_landmarks or []:
            pts = np.array([(p.x * W, p.y * H) for p in lm.landmark])
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            size = max(x1 - x0, y1 - y0)
            opening = abs(pts[14, 1] - pts[13, 1]) / max(y1 - y0, 1.0)  # môi trong trên/dưới / chiều cao mặt
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            m = 0.15 * size
            crop = rgb[int(max(y0 - m, 0)):int(min(y1 + m, H)), int(max(x0 - m, 0)):int(min(x1 + m, W))]
            if crop.size == 0:
                continue
            best, bd = None, 0.6 * size
            for t in tracks:
                d = np.linalg.norm(t["c"] - c)
                if i not in t["frames"] and d < bd:
                    best, bd = t, d
            if best is None:
                best = {"c": c, "frames": set(), "open": [], "size": [], "img": []}
                tracks.append(best)
            best["c"] = c
            best["frames"].add(i)
            best["open"].append(opening)
            best["size"].append(size)
            best["img"].append(Image.fromarray(crop))
    cap.release()
    fm.close()
    if not tracks:
        return []
    moving = [t for t in tracks if len(t["open"]) >= 3]
    t = max(moving, key=lambda t: np.std(t["open"])) if moving else max(tracks, key=lambda t: np.mean(t["size"]))
    return pick_frames(t["img"], n_out)


# ----------------------------------------------------------------------------- DiGEMO
def load_digemo(a, dev):
    from model import DiGemo

    args = SimpleNamespace(no_cuda=dev == "cpu", win=a.win, modals=a.modals, fusion_method=a.fusion_method,
                           no_residual=a.no_residual, no_graph=a.no_graph, no_intra=a.no_intra,
                           hidden_dim=a.hidden_dim, num_heads=a.num_heads, dropout_1=a.dropout_1,
                           dropout_2=a.dropout_2, heter_n_layers=a.heter_n_layers)
    m = DiGemo(args, [1024, 342, 300], 7)
    ck = torch.load(a.digemo, map_location="cpu")
    sd = ck
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(ck, dict) and isinstance(ck.get(k), dict):
            sd = ck[k]
            break
    m.load_state_dict(sd)
    return m.to(dev).eval()


def transcribe(paths, size="small"):
    from faster_whisper import WhisperModel

    w = WhisperModel(size, device="cuda" if torch.cuda.is_available() else "cpu")
    return [" ".join(s.text.strip() for s in w.transcribe(p, language="en")[0]) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--digemo", required=True, help="checkpoint DiGEMO đã train trên pkl mới")
    ap.add_argument("--text_ckpt", default="text_ft/best.pt")
    ap.add_argument("--face_ckpt", default="ckpt_fer/best.pt")
    ap.add_argument("--feat_dir", default="feats", help="chứa audio_head.npz, norm.npz, speaker_map.json")
    ap.add_argument("--w2v", default="facebook/wav2vec2-base")
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--out", default="pred.csv")
    # siêu tham số DiGEMO: PHẢI trùng lúc train
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--dropout_1", type=float, default=0.1)
    ap.add_argument("--dropout_2", type=float, default=0.1)
    ap.add_argument("--win", type=int, nargs=2, default=[10, 10])
    ap.add_argument("--heter_n_layers", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--modals", default="tva")
    ap.add_argument("--fusion_method", default="gated")
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--no_graph", action="store_true")
    ap.add_argument("--no_intra", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.csv)
    df["key"] = [key(d, u) for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])]
    files = collect_files(a.video_dir)
    miss = [k for k in df["key"] if k not in files]
    if miss:
        raise SystemExit(f"thiếu {len(miss)} video, ví dụ: {miss[:5]}")
    paths = [files[k] for k in df["key"]]

    # --- text
    if "Utterance" not in df or df["Utterance"].isna().any() or (df["Utterance"].astype(str).str.strip() == "").any():
        print("csv thiếu câu thoại -> faster-whisper")
        df["Utterance"] = transcribe(paths)
    from transformers import AutoTokenizer

    tm = TextEncoder.load(a.text_ckpt, dev)
    ft = text_features(tm, AutoTokenizer.from_pretrained(tm.base), df["Utterance"].astype(str).tolist(), dev)
    del tm

    # --- audio
    ae = AudioEncoder(a.w2v).to(dev)
    ae.load_head(os.path.join(a.feat_dir, "audio_head.npz"))
    fa = ae.project(ae.raw768(paths, bs=8))
    del ae

    # --- visual
    norm = np.load(os.path.join(a.feat_dir, "norm.npz"))
    fe = FaceEncoder(a.face_ckpt, dev)
    fv = np.zeros((len(df), 342), np.float32)  # thiếu mặt -> 0 (= trung bình train sau chuẩn hóa)
    for i, p in enumerate(paths):
        crops = speaker_crops(p, n_out=a.nframes)
        if crops:
            fv[i] = (fe.embed_pil(crops).mean(0) - norm["v_mean"]) / norm["v_std"]
        else:
            print(f"[WARN] không thấy mặt: {p}")

    # --- speaker one-hot 9 chiều
    spk = json.load(open(os.path.join(a.feat_dir, "speaker_map.json"), encoding="utf-8"))
    sp_col = df["Speaker"].astype(str) if "Speaker" in df else pd.Series(["__other__"] * len(df))
    sidx = np.array([spk.get(s, spk["__other__"]) for s in sp_col])

    # --- DiGEMO: đưa cả hội thoại vào cùng lúc
    model = load_digemo(a, dev)
    pred = np.zeros(len(df), int)
    with torch.no_grad():
        for dia, g in df.sort_values(["Dialogue_ID", "Utterance_ID"]).groupby("Dialogue_ID", sort=False):
            ix = g.index.to_numpy()
            L = len(ix)
            t = torch.tensor(ft[ix]).unsqueeze(1).to(dev)  # [L,1,1024]
            v = torch.tensor(fv[ix]).unsqueeze(1).to(dev)
            au = torch.tensor(fa[ix]).unsqueeze(1).to(dev)
            q = torch.zeros(L, 1, 9)
            q[torch.arange(L), 0, torch.tensor(sidx[ix])] = 1
            umask = torch.ones(L, 1).to(dev)
            logit = model(t, v, au, umask, q.to(dev), [L])[0]
            pred[ix] = logit.argmax(-1).cpu().numpy()

    df["pred_id"] = pred
    df["pred_emotion"] = [EMO_NAMES[i] for i in pred]
    df.drop(columns=["key"]).to_csv(a.out, index=False, encoding="utf-8-sig")
    print("saved ->", a.out)
    if "Emotion" in df:
        from sklearn.metrics import accuracy_score, f1_score

        y = [EMO_NAMES.index(str(e).lower()) for e in df["Emotion"]]
        print(f"acc {accuracy_score(y, pred):.4f}  weighted-F1 {f1_score(y, pred, average='weighted'):.4f}")


if __name__ == "__main__":
    main()
