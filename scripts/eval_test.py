
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'digemo' / d) for d in ('encoders', 'model', 'pipeline')]

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from common import EMO, EMO_NAMES, collect_files, load_pkl, natural_key, read_split_csv, save_pkl

DIMS = {"text": 1024, "audio": 300, "visual": 342}
IEMOCAP_NAMES = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]  # thứ tự nhãn DiGemo IEMOCAP
NAMES = list(EMO_NAMES)   # đổi thành IEMOCAP_NAMES khi checkpoint là IEMOCAP
DATASET = "MELD"


def in_root(p):
    if p is None:
        return None
    p = Path(p).expanduser()
    return p if p.is_absolute() else R / p


# ============================================================================ đọc frame / mặt
def video_duration(cap):
    import cv2
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    return float(n / fps) if fps > 0 else 0.0


def full_frames(cap, n_out=8):
    import cv2
    from PIL import Image
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for i in np.unique(np.linspace(0, max(total - 1, 0), n_out).round().astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    return out


def load_face_json(path, base, nframes):
    from encoders import pick_frames
    with open(path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    base = base or os.path.dirname(os.path.abspath(path))
    return {k: pick_frames(sorted([os.path.join(base, p) for p in v], key=natural_key), nframes)
            for k, v in groups.items() if v}


# ============================================================================ stage extract
def fill_vector(feat_dir, name, dim):
    """giống build_features: thiếu feature -> trung bình train (nếu có pkl train), không thì 0."""
    p = feat_dir / f"{name}_train.pkl"
    if p.exists():
        return np.stack(list(load_pkl(str(p)).values())).mean(0).astype(np.float32)
    return np.zeros(dim, np.float32)


def stage_extract(a, df, dev):
    from encoders import AudioEncoder, FaceEncoder, TextEncoder, text_features

    cache_path = a.out_dir / f"test_feats_{a.face_mode}.pkl"
    cache = load_pkl(str(cache_path)) if cache_path.exists() and not a.force else {}
    keys = df["key"].tolist()
    files = collect_files(str(a.video_dir), exts=(".mp4", ".avi", ".mkv", ".mov"))
    missing = [k for k in keys if k not in files]
    print(f"[extract] {len(keys)} câu, {len(missing)} thiếu video, {len(cache)} đã có trong cache")
    if missing:
        print("   ví dụ thiếu:", missing[:5])

    def entry(k):
        return cache.setdefault(k, {"missing_video": k not in files})

    # ---- text (nhanh, chạy một lượt)
    todo = [k for k in keys if "text" not in cache.get(k, {})]
    if todo:
        from transformers import AutoTokenizer
        tm = TextEncoder.load(str(a.text_ckpt), dev)
        tok = AutoTokenizer.from_pretrained(tm.base)
        txt = df.set_index("key").loc[todo, "Utterance"].astype(str).tolist()
        f = text_features(tm, tok, txt, dev, a.bs_text)
        for k, v in zip(todo, f):
            entry(k)["text"] = v
        del tm
        save_pkl(cache, str(cache_path))
        print(f"[text] {len(todo)} câu")

    # ---- audio: y hệt build_features (decode cả file mp4, tối đa 15 s, Wav2Vec2 mean -> audio_head)
    todo = [k for k in keys if "audio" not in cache.get(k, {})]
    if todo:
        ae = AudioEncoder(a.w2v).to(dev)
        ae.load_head(str(a.feat_dir / "audio_head.npz"))
        have = [k for k in todo if k in files]
        fill = fill_vector(a.feat_dir, "audio_300", 300)
        for i in range(0, len(have), a.chunk):
            part = have[i:i + a.chunk]
            proj = ae.project(ae.raw768([files[k] for k in part], a.bs_audio, a.workers))
            for k, v in zip(part, proj):
                entry(k)["audio"] = v
            save_pkl(cache, str(cache_path))
            print(f"[audio] {min(i + a.chunk, len(have))}/{len(have)}")
        for k in todo:
            if k not in files:
                entry(k)["audio"] = fill
        del ae
        save_pkl(cache, str(cache_path))

    # ---- visual
    todo = [k for k in keys if "visual" not in cache.get(k, {})]
    if todo:
        fe = FaceEncoder(str(a.face_ckpt), dev)
        norm = np.load(a.feat_dir / "norm.npz")
        v_mean, v_std = norm["v_mean"], norm["v_std"]
        cropper, groups = None, None
        if a.face_mode == "speaker":
            from encoders_online import SpeakerFaceCropper
            cropper = SpeakerFaceCropper()
        elif a.face_mode == "json":
            groups = load_face_json(str(a.face_json), a.face_base, a.nframes)
        import cv2
        from PIL import Image
        try:
            for i, k in enumerate(todo):
                imgs = []
                if a.face_mode == "json":
                    imgs = [Image.open(p).convert("RGB") for p in groups.get(k, [])]
                elif k in files:
                    cap = cv2.VideoCapture(files[k])
                    try:
                        if a.face_mode == "speaker":
                            imgs = cropper(cap, 0.0, max(video_duration(cap), 0.1), n_out=a.nframes)
                        else:
                            imgs = full_frames(cap, a.nframes)
                    finally:
                        cap.release()
                e = entry(k)
                e["no_face"] = not imgs
                # không có mặt -> 0 (= trung bình train sau chuẩn hóa), như build_features / web
                e["visual"] = (((fe.embed_pil(imgs).mean(0) - v_mean) / v_std).astype(np.float32)
                               if imgs else np.zeros(342, np.float32))
                if (i + 1) % a.chunk == 0:
                    save_pkl(cache, str(cache_path))
                    print(f"[visual/{a.face_mode}] {i + 1}/{len(todo)}")
        finally:
            if cropper is not None:
                cropper.close()
            save_pkl(cache, str(cache_path))
        nf = sum(int(cache[k].get("no_face", False)) for k in keys)
        print(f"[visual/{a.face_mode}] xong, {nf} câu không thấy mặt")

    for k in keys:
        for name, dim in DIMS.items():
            if np.shape(cache[k][name]) != (dim,):
                raise RuntimeError(f"{k}: {name} sai kích thước {np.shape(cache[k][name])}")
    print("[extract] cache ->", cache_path)
    return cache


# ============================================================================ stage predict
def speaker_slots(speakers, feat_dir):
    raw = json.load(open(feat_dir / "speaker_map.json", encoding="utf-8"))
    other = raw.get("__other__", 8)
    exact = {k: v for k, v in raw.items() if k != "__other__"}
    lower = {k.strip().lower(): v for k, v in exact.items()}
    return [exact.get(s, lower.get(str(s).strip().lower(), other)) for s in speakers]


def dialogues_from_raw(df, cache, feat_dir):
    """-> list of dict(rows=DataFrame, t, v, a, slots)"""
    out = []
    for dia, g in df.sort_values(["Dialogue_ID", "Utterance_ID"]).groupby("Dialogue_ID", sort=True):
        ks = g["key"].tolist()
        out.append({
            "rows": g,
            "t": np.stack([cache[k]["text"] for k in ks]),
            "v": np.stack([cache[k]["visual"] for k in ks]),
            "a": np.stack([cache[k]["audio"] for k in ks]),
            "slots": speaker_slots(g["Speaker"].astype(str).tolist(), feat_dir),
        })
    return out


def dialogues_from_pkl(a):
    """testVid của pkl.
    MELD (14 phần tử, bài báo hoặc build_features.py): 0 IDs,1 Spk one-hot,2 Lab,4 Txt,8 Aud,9 Vis,10 Sent,12 testVid
    IEMOCAP (12 phần tử, bài báo): 0 IDs,1 Spk 'M'/'F',2 Lab,3 Txt,7 Aud(1582),8 Vis,9 Sent,11 testVid
    """
    import pickle
    with open(a.pkl, "rb") as f:
        try:
            data = pickle.load(f)
        except UnicodeDecodeError:  # pkl cũ của bài báo (Python 2)
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    if len(data) == 12:
        IDs, Spk, Lab, Txt, Aud, Vis, Sent, test_vid = (data[0], data[1], data[2], data[3], data[7], data[8],
                                                        data[9], data[11])
    else:
        IDs, Spk, Lab, Txt, Aud, Vis, Sent, test_vid = (data[0], data[1], data[2], data[4], data[8], data[9],
                                                        data[10], data[12])
    inv = {}
    km = a.pkl.parent / "key_map.json"
    if km.exists():
        inv = {v: int(k.split("/")[1]) for k, v in json.load(open(km))["map"].items() if k.startswith("test/")}
    out = []
    for vid in test_vid:
        n = len(Lab[vid])
        rows = pd.DataFrame({
            "Dialogue_ID": [inv.get(vid, vid)] * n,
            "Utterance_ID": list(IDs[vid]),
            "Speaker": [str(s) for s in Spk[vid]] if len(data) == 12 else [""] * n,
            "Utterance": list(Sent[vid]),
            "Emotion": [NAMES[int(y)] for y in Lab[vid]],
        })
        if len(data) == 12:   # như dataloader IEMOCAPDataset_BERT: M -> [1,0], F -> [0,1]
            slots = [0 if str(s) == "M" else 1 for s in Spk[vid]]
        else:
            slots = np.asarray(Spk[vid]).argmax(1).tolist()
        out.append({"rows": rows, "t": np.asarray(Txt[vid]), "v": np.asarray(Vis[vid]),
                    "a": np.asarray(Aud[vid]), "slots": slots})
    return out


@torch.no_grad()
def run_model(model, dialogues, n_speakers, dev):
    probs = []
    for d in dialogues:
        L = len(d["slots"])
        t = torch.tensor(d["t"], dtype=torch.float32).unsqueeze(1).to(dev)
        v = torch.tensor(d["v"], dtype=torch.float32).unsqueeze(1).to(dev)
        au = torch.tensor(d["a"], dtype=torch.float32).unsqueeze(1).to(dev)
        q = torch.zeros(L, 1, n_speakers, device=dev)
        q[torch.arange(L), 0, torch.tensor(d["slots"])] = 1.0
        umask = torch.ones(L, 1, device=dev)
        logit = model(t, v, au, umask, q, [L])[0]
        probs.append(torch.softmax(logit, -1).float().cpu().numpy())
    return probs


def report(a, table):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from predict import plot_confusion_matrix_styled

    lab = table.dropna(subset=["Emotion"])
    idx = {n: i for i, n in enumerate(NAMES)}
    C = len(NAMES)
    lab = lab[lab["Emotion"].astype(str).str.lower().isin(idx)]
    if lab.empty:
        print("[report] không có nhãn Emotion -> bỏ qua metrics")
        return
    y = lab["Emotion"].astype(str).str.lower().map(idx).to_numpy()
    p = lab["pred_id"].to_numpy()
    rep = classification_report(y, p, labels=list(range(C)), target_names=NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(C)))
    m = {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
         "weighted_f1": float(f1_score(y, p, average="weighted")),
         "macro_f1": float(f1_score(y, p, average="macro")),
         "per_class_f1": dict(zip(NAMES, f1_score(y, p, labels=list(range(C)), average=None,
                                                      zero_division=0).round(4).tolist())),
         "source": a.source, "face_mode": a.face_mode if a.source == "raw" else "pkl"}
    print(rep)
    print(f"acc {m['accuracy']:.4f}  weighted-F1 {m['weighted_f1']:.4f}  macro-F1 {m['macro_f1']:.4f}")
    (a.out_dir / "metrics.txt").write_text(rep + f"\nacc {m['accuracy']:.4f}  weighted-F1 "
                                           f"{m['weighted_f1']:.4f}  macro-F1 {m['macro_f1']:.4f}\n",
                                           encoding="utf-8")
    json.dump(m, open(a.out_dir / "metrics.json", "w"), indent=1)
    pd.DataFrame(cm, index=NAMES, columns=NAMES).to_csv(a.out_dir / "confusion_counts.csv")
    for fmt in ("pdf", "png"):
        path = plot_confusion_matrix_styled(cm, DATASET, f"cm_{DATASET.lower()}", save_dir=str(a.out_dir), img_format=fmt)
        print("[report] ->", path)


def stage_predict(a, df, cache, dev):
    from predict import load_model_from_checkpoint

    global NAMES, DATASET
    model, cfg, dims, n_classes, ns, dev = load_model_from_checkpoint(str(a.checkpoint), dev)
    DATASET = getattr(cfg, "dataset", "MELD")
    if DATASET == "IEMOCAP":
        if a.source != "pkl":
            raise SystemExit("Checkpoint IEMOCAP chỉ chạy được với --source pkl (encoder DiGEmo là của MELD)")
        NAMES = list(IEMOCAP_NAMES)
    elif list(dims) != [1024, 342, 300] or n_classes != 7:
        raise SystemExit(f"Checkpoint không phải MELD [1024,342,300]/7 nhãn: {dims}, {n_classes}")
    print(f"[checkpoint] {DATASET} dims={dims} classes={n_classes} speakers={ns}")
    dialogues = dialogues_from_pkl(a) if a.source == "pkl" else dialogues_from_raw(df, cache, a.feat_dir)
    probs = run_model(model, dialogues, ns, dev)

    parts = []
    for d, pr in zip(dialogues, probs):
        g = d["rows"][[c for c in ("Dialogue_ID", "Utterance_ID", "Speaker", "Utterance", "Emotion")
                       if c in d["rows"]]].copy()
        g["slot"] = d["slots"]
        g["pred_id"] = pr.argmax(1)
        g["Predicted"] = [NAMES[i] for i in g["pred_id"]]
        g["Confidence"] = pr.max(1).round(4)
        for j, n in enumerate(NAMES):
            g[f"prob_{n}"] = pr[:, j].round(4)
        if cache is not None and "key" in d["rows"]:
            g["no_face"] = [bool(cache[k].get("no_face", False)) for k in d["rows"]["key"]]
            g["missing_video"] = [bool(cache[k].get("missing_video", False)) for k in d["rows"]["key"]]
        parts.append(g)
    table = pd.concat(parts, ignore_index=True)
    out = a.out_dir / "predictions.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[predict] {len(table)} câu, {len(dialogues)} hội thoại -> {out}")
    report(a, table)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["raw", "pkl"], default="raw",
                    help="raw: csv + video thô qua encoder; pkl: testVid của pkl train (mốc so sánh)")
    ap.add_argument("--stages", nargs="+", default=["extract", "predict"], choices=["extract", "predict"])
    ap.add_argument("--csv", help="test_sent_emo.csv (source raw)")
    ap.add_argument("--video_dir", help="thư mục mp4 test (diaX_uttY.mp4), quét đệ quy")
    ap.add_argument("--pkl", help="pkl 14 phần tử từ build_features.py (source pkl)")
    ap.add_argument("--out_dir", default="results/test_eval")
    ap.add_argument("--checkpoint", default="checkpoints/digemo_meld.pth")
    ap.add_argument("--text_ckpt", default="checkpoints/text_ft.pt")
    ap.add_argument("--face_ckpt", default="checkpoints/face_fer.pt")
    ap.add_argument("--feat_dir", default="data/feats", help="audio_head.npz, norm.npz, speaker_map.json")
    ap.add_argument("--w2v", default="facebook/wav2vec2-base")
    ap.add_argument("--face_mode", choices=["speaker", "full", "json"], default="speaker")
    ap.add_argument("--face_json", help="FacSeqs test json (face_mode json)")
    ap.add_argument("--face_base", default="", help="gốc ghép với đường dẫn trong face_json")
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--bs_text", type=int, default=64)
    ap.add_argument("--bs_audio", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200, help="lưu cache sau mỗi chừng này câu")
    ap.add_argument("--limit_dialogues", type=int, default=0, help="chỉ chạy N hội thoại đầu (chạy thử)")
    ap.add_argument("--force", action="store_true", help="bỏ cache, trích lại từ đầu")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    for n in ("out_dir", "checkpoint", "text_ckpt", "face_ckpt", "feat_dir", "pkl", "csv", "video_dir", "face_json"):
        setattr(a, n, in_root(getattr(a, n)))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print("[device]", dev)

    df, cache = None, None
    if a.source == "raw":
        if not a.csv:
            ap.error("--source raw cần --csv")
        df = read_split_csv(str(a.csv))
        if a.limit_dialogues:
            keep = sorted(df["Dialogue_ID"].unique())[:a.limit_dialogues]
            df = df[df["Dialogue_ID"].isin(keep)]
        if "extract" in a.stages:
            if not a.video_dir:
                ap.error("cần --video_dir (audio luôn lấy từ mp4)")
            if a.face_mode == "json" and not a.face_json:
                ap.error("--face_mode json cần --face_json")
            cache = stage_extract(a, df, dev)
        else:
            cp = a.out_dir / f"test_feats_{a.face_mode}.pkl"
            if not cp.exists():
                raise SystemExit(f"Chưa có cache {cp}: chạy stage extract trước")
            cache = load_pkl(str(cp))
    elif not a.pkl:
        ap.error("--source pkl cần --pkl")

    if "predict" in a.stages:
        stage_predict(a, df, cache, dev)


if __name__ == "__main__":
    main()
