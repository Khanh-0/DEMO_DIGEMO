"""Chạy 3 encoder cho train/val/test rồi DỰNG pkl 14 phần tử (đúng định dạng MELDDataset_BERT) trực tiếp từ csv MELD.
Không cần pkl của paper => dataloader/model DiGEMO vẫn giữ nguyên.

Các stage (có cache, chạy lại bỏ qua phần đã có, --force để làm lại):
    text   : text_1024_{split}.pkl       cần --text_ckpt (từ text_finetune.py) + 3 csv
    audio  : audio_300_{split}.pkl, audio_head.npz   cần 3 thư mục mp4 (PCA fit trên train)
    visual : face_342_{split}.pkl        cần --face_ckpt + 3 json FacSeqs (--nframes 8 như paper)
    merge  : dựng --out từ csv + 3 modal; xuất norm.npz (scaler visual), speaker_map.json, key_map.json

pkl: 0 IDs, 1 Speakers(one-hot 9), 2 Labels, 3 Sentiments, 4-7 Text0..3 (1024), 8 Audio (300), 9 Visual (342),
     10 Sentence, 11 trainVid (= train + val, như paper), 12 testVid, 13 None.
Key hội thoại: đánh số liên tiếp train -> val -> test (id của MELD bị trùng giữa các split nên phải đánh lại).

Ví dụ:
    python build_features.py --out_dir feats --out features\\meld_new.pkl ^
      --text_ckpt text_ft\\best.pt --train_csv train_sent_emo.csv --val_csv dev_sent_emo.csv --test_csv test_sent_emo.csv ^
      --train_mp4 <train_splits> --val_mp4 E:\\Do_an\\Dataprocessing\\MELD.Raw\\dev\\dev_splits_complete --test_mp4 <test_splits> ^
      --face_ckpt ckpt_fer\\best.pt --train_json <train.json> --val_json <val.json> --test_json <test.json> --face_base <gốc FacSeqs>
"""
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'digemo' / d) for d in ('encoders', 'model', 'pipeline')]

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from common import EMO, collect_files, load_pkl, natural_key, read_split_csv, save_pkl
from encoders import AudioEncoder, FaceEncoder, TextEncoder, fold_pca, pick_frames, text_features

SPL = ("train", "val", "test")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def P(a, name, sp):
    return os.path.join(a.out_dir, f"{name}_{sp}.pkl")


def todo(a, name):
    return [sp for sp in SPL if a.force or not os.path.exists(P(a, name, sp))]


# ----------------------------------------------------------------------------- text
def stage_text(a):
    sps = todo(a, "text_1024")
    if not sps:
        return
    from transformers import AutoTokenizer

    model = TextEncoder.load(a.text_ckpt, DEV)
    tok = AutoTokenizer.from_pretrained(model.base)
    for sp in sps:
        df = read_split_csv(getattr(a, f"{sp}_csv"))
        f = text_features(model, tok, df["Utterance"].tolist(), DEV, a.bs_text)
        save_pkl(dict(zip(df["key"], f)), P(a, "text_1024", sp))
        print(f"[text/{sp}] {len(df)} câu")


# ----------------------------------------------------------------------------- audio
def stage_audio(a):
    head = os.path.join(a.out_dir, "audio_head.npz")
    if not todo(a, "audio_300") and os.path.exists(head) and not a.force:
        return
    enc, raw = None, {}
    for sp in SPL:
        cache = P(a, "raw768", sp)
        if os.path.exists(cache) and not a.force:
            raw[sp] = load_pkl(cache)
            continue
        enc = enc or AudioEncoder(a.w2v).to(DEV)
        files = collect_files(getattr(a, f"{sp}_mp4"))
        ks = sorted(files)
        print(f"[audio/{sp}] {len(ks)} file")
        arr = enc.raw768([files[k] for k in ks], a.bs_audio, a.workers)
        raw[sp] = dict(zip(ks, arr))
        save_pkl(raw[sp], cache)
    if not os.path.exists(head) or a.force:
        W, b = fold_pca(np.stack(list(raw["train"].values())))
        np.savez(head, W=W, b=b)
    d = np.load(head)
    for sp in SPL:
        save_pkl({k: (v @ d["W"].T + d["b"]).astype(np.float32) for k, v in raw[sp].items()}, P(a, "audio_300", sp))


# ----------------------------------------------------------------------------- visual
def stage_visual(a):
    sps = todo(a, "face_342")
    if not sps:
        return
    enc = FaceEncoder(a.face_ckpt, DEV)
    for sp in sps:
        with open(getattr(a, f"{sp}_json"), "r", encoding="utf-8") as f:
            groups = json.load(f)
        groups = {k: pick_frames(sorted([os.path.join(a.face_base, p) for p in v], key=natural_key), a.nframes)
                  for k, v in groups.items() if v}
        flat = [p for ps in groups.values() for p in ps]
        print(f"[visual/{sp}] {len(groups)} câu, {len(flat)} frame")
        emb = enc.embed_paths(flat, a.bs_face, a.workers)
        out, pos = {}, 0
        for k, ps in groups.items():
            out[k] = emb[pos:pos + len(ps)].mean(0)
            pos += len(ps)
        save_pkl(out, P(a, "face_342", sp))


# ----------------------------------------------------------------------------- merge (dựng pkl từ csv)
SENT = {"neutral": 0, "positive": 1, "negative": 2}  # DiGEMO không dùng videoSentiments


def stage_merge(a):
    dfs = {sp: read_split_csv(getattr(a, f"{sp}_csv")) for sp in SPL}

    # 9 người nói: 8 nhân vật xuất hiện nhiều nhất trong train + 1 nhóm "khác"
    top = [n for n, _ in Counter(dfs["train"]["Speaker"].astype(str)).most_common(8)]
    spk = {n: i for i, n in enumerate(top)}
    spk["__other__"] = 8
    with open(os.path.join(a.out_dir, "speaker_map.json"), "w", encoding="utf-8") as f:
        json.dump(spk, f, ensure_ascii=False, indent=1)
    print("[speaker]", top, "+ khác")

    feats = {n: {sp: load_pkl(P(a, n, sp)) for sp in SPL} for n in ("text_1024", "audio_300", "face_342")}
    tv = np.stack(list(feats["face_342"]["train"].values()))
    v_mean, v_std = tv.mean(0), np.maximum(tv.std(0), 1e-6)  # chuẩn hóa visual bằng thống kê train
    np.savez(os.path.join(a.out_dir, "norm.npz"), v_mean=v_mean, v_std=v_std)
    fill = {"text_1024": None, "audio_300": None, "face_342": v_mean}
    for n in ("text_1024", "audio_300"):
        fill[n] = np.stack(list(feats[n]["train"].values())).mean(0)

    IDs, Spk, Lab, Sen, Txt, Aud, Vis, Sent = ({} for _ in range(8))
    vids = {sp: [] for sp in SPL}
    key_map, vid, miss = {}, 0, {n: 0 for n in fill}
    for sp in SPL:
        df = dfs[sp].sort_values(["Dialogue_ID", "Utterance_ID"])
        for dia, g in df.groupby("Dialogue_ID", sort=True):
            ks = g["key"].tolist()
            IDs[vid] = g["Utterance_ID"].astype(int).tolist()
            Spk[vid] = np.eye(9, dtype=np.float32)[[spk.get(str(s), 8) for s in g["Speaker"]]]
            Lab[vid] = [EMO[str(e).lower()] for e in g["Emotion"]]
            Sen[vid] = [SENT.get(str(s).lower(), 0) for s in g["Sentiment"]] if "Sentiment" in g else [0] * len(ks)
            Sent[vid] = g["Utterance"].tolist()
            for name, dst in (("text_1024", Txt), ("audio_300", Aud), ("face_342", Vis)):
                rows = []
                for k in ks:
                    v = feats[name][sp].get(k)
                    if v is None:
                        miss[name] += 1
                        v = fill[name]
                    rows.append(v)
                arr = np.stack(rows).astype(np.float32)
                dst[vid] = (arr - v_mean) / v_std if name == "face_342" else arr
            vids[sp].append(vid)
            key_map[f"{sp}/{int(dia)}"] = vid
            vid += 1
    total = sum(len(v) for d in dfs.values() for v in [d])
    print(f"[merge] {vid} hội thoại, {total} câu; thiếu feature: " + ", ".join(f"{n}={c}" for n, c in miss.items()))
    print("   hội thoại: " + ", ".join(f"{sp}={len(v)}" for sp, v in vids.items()))

    train_vid = vids["train"] + ([] if a.no_val_in_train else vids["val"])
    data = (IDs, Spk, Lab, Sen, Txt, Txt, Txt, Txt, Aud, Vis, Sent, train_vid, vids["test"], None)
    with open(os.path.join(a.out_dir, "key_map.json"), "w") as f:
        json.dump({"val_vids": vids["val"], "map": key_map}, f)
    save_pkl(data, a.out)
    print("saved ->", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="feats")
    ap.add_argument("--stages", nargs="+", default=["text", "audio", "visual", "merge"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_val_in_train", action="store_true", help="trainVid chỉ gồm train (mặc định gồm train+val như paper)")
    ap.add_argument("--out")
    ap.add_argument("--text_ckpt")
    ap.add_argument("--face_ckpt")
    ap.add_argument("--w2v", default="facebook/wav2vec2-base")
    ap.add_argument("--face_base", default="")
    ap.add_argument("--nframes", type=int, default=8, help="8 frame cách đều/câu như paper; 0 = dùng hết")
    for s in SPL:
        ap.add_argument(f"--{s}_csv")
        ap.add_argument(f"--{s}_mp4")
        ap.add_argument(f"--{s}_json")
    ap.add_argument("--bs_text", type=int, default=64)
    ap.add_argument("--bs_audio", type=int, default=8)
    ap.add_argument("--bs_face", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for st in a.stages:
        {"text": stage_text, "audio": stage_audio, "visual": stage_visual, "merge": stage_merge}[st](a)


if __name__ == "__main__":
    main()