"""So sánh DiGEmo và DM-CER trên tập TEST của MELD hoặc IEMOCAP (một file duy nhất, thay cho
compare_iemocap.py và phần --source pkl của eval_test.py).

Nhãn thật  : lấy từ pkl gốc (testVid) mà DiGEmo đã dùng để train/test.
DiGEmo     : checkpoint DiGEmo chạy trên đặc trưng testVid của pkl (hoặc đọc predictions.csv có sẵn).
DM-CER     : đọc file dự đoán của DM-CER (MELD: csv có Dialogue_ID, Utterance_ID, prediction;
             IEMOCAP: tsv có utt_id, prediction), hoặc tự gọi inference.py của repo DM-CER (--run_dmcer).

Ánh xạ nhãn (viết tường minh, không đoán):
  MELD    DiGEmo/pkl id : 0 neutral,1 surprise,2 fear,3 sadness,4 joy,5 disgust,6 anger
          DM-CER id     : 0 neutral,1 joy,2 sadness,3 anger,4 surprise,5 fear,6 disgust
  IEMOCAP DiGEmo/pkl id : 0 hap,1 sad,2 neu,3 ang,4 exc,5 fru
          DM-CER id     : 0 neu,1 fru,2 ang,3 sad,4 hap,5 exc
Mọi thứ được đổi sang TÊN nhãn rồi mới so sánh; bảng/ma trận hiển thị theo thứ tự DISPLAY bên dưới.

Đầu ra trong <out_dir>/<dataset>/:
  per_utterance.csv   mỗi câu: id, nhãn thật, DiGEmo, DM-CER, đúng/sai từng model
  disagreements.csv   các câu hai model dự đoán khác nhau
  metrics.json        acc, weighted-F1, macro-F1, F1/precision/recall từng lớp, ma trận nhầm lẫn, độ trùng khớp
  summary.md          bảng tóm tắt
  cm_compare.png      hai ma trận nhầm lẫn cạnh nhau;  f1_per_class.png  F1 từng lớp
  compare.json        toàn bộ kết quả (web đọc file này)

Cấu hình đường dẫn: configs/compare.json (tạo mẫu bằng --write_config), tham số dòng lệnh ghi đè.

Ví dụ (PowerShell, tại gốc demo_meld, đã activate .venv):
  python scripts\\compare_models.py --write_config            # tạo configs\\compare.json rồi sửa đường dẫn
  python scripts\\compare_models.py --dataset meld
  python scripts\\compare_models.py --dataset iemocap --run_dmcer
  python scripts\\compare_models.py --dataset meld --dmcer_pred D:\\...\\pred_test.csv --pkl data\\feats\\meld_multi_features.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]          # gốc demo_meld
DEFAULT_CONFIG = R / "configs" / "compare.json"

DATASETS = ("meld", "iemocap")
DISPLAY = {
    "meld": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
    "iemocap": ["neu", "fru", "ang", "sad", "hap", "exc"],
}
DIGEMO_ID2LAB = {
    "meld": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
    "iemocap": ["hap", "sad", "neu", "ang", "exc", "fru"],
}
DMCER_ID2LAB = {
    "meld": ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"],
    "iemocap": ["neu", "fru", "ang", "sad", "hap", "exc"],
}
ALIAS = {
    "meld": {"happy": "joy", "happiness": "joy", "sad": "sadness", "angry": "anger", "neu": "neutral",
             "surprised": "surprise", "fearful": "fear", "disgusted": "disgust"},
    "iemocap": {"happy": "hap", "happiness": "hap", "sad": "sad", "sadness": "sad", "neutral": "neu",
                "angry": "ang", "anger": "ang", "excited": "exc", "frustrated": "fru", "frustration": "fru"},
}

TEMPLATE = {
    "out_dir": "results/compare",
    "device": "auto",
    "meld": {
        "pkl": "data/feats/meld_multi_features.pkl",
        "csv": "data/csv/test_sent_emo.csv",
        "key_map": "",
        "digemo_ckpt": "checkpoints/digemo_meld.pth",
        "digemo_pred": "",
        "dmcer_pred": "results/dmcer/meld_pred_test.csv",
        "dmcer_code_dir": "",
        "dmcer_data_root": "",
        "dmcer_ckpt": "",
    },
    "iemocap": {
        "pkl": "data/feats/iemocap_multi_features.pkl",
        "key_map": "",
        "digemo_ckpt": "checkpoints/digemo_iemocap.pth",
        "digemo_pred": "",
        "dmcer_pred": "results/dmcer/iemocap_pred_test.tsv",
        "dmcer_code_dir": "",
        "dmcer_data_root": "",
        "dmcer_ckpt": "",
    },
}
PATH_KEYS = ("pkl", "csv", "key_map", "digemo_ckpt", "digemo_pred", "dmcer_pred", "dmcer_code_dir",
             "dmcer_data_root", "dmcer_ckpt")


# ============================================================================ config
def in_root(p):
    if not p:
        return None
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else R / p


def load_config(path=None):
    """configs/compare.json trộn với TEMPLATE (thiếu khóa nào thì lấy mặc định)."""
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = json.loads(json.dumps(TEMPLATE))
    if path.is_file():
        user = json.loads(path.read_text(encoding="utf-8"))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def describe_config(cfg):
    """đường dẫn tuyệt đối + có tồn tại hay không (web hiển thị)."""
    out = {"out_dir": str(in_root(cfg["out_dir"]))}
    for ds in DATASETS:
        d = {}
        for k in PATH_KEYS:
            p = in_root(cfg[ds].get(k))
            d[k] = {"path": str(p) if p else "", "exists": bool(p and p.exists())}
        d["can_run_dmcer"] = all(d[k]["exists"] for k in ("dmcer_code_dir", "dmcer_data_root", "dmcer_ckpt"))
        d["can_run_digemo"] = d["pkl"]["exists"] and (d["digemo_ckpt"]["exists"] or d["digemo_pred"]["exists"])
        out[ds] = d
    return out


# ============================================================================ nhãn
def norm_label(ds, x, id2lab=None):
    """chuỗi/số -> tên nhãn chuẩn của DISPLAY[ds]; None nếu không nhận ra."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    if id2lab is not None and s.lstrip("-").isdigit():
        i = int(s)
        return id2lab[i] if 0 <= i < len(id2lab) else None
    s = s.lower()
    s = ALIAS[ds].get(s, s)
    return s if s in DISPLAY[ds] else None


# ============================================================================ nhãn thật + DiGEmo (pkl)
def read_pkl(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:   # pkl Python 2 của bài báo
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def _norm_text(s):
    import re
    s = str(s).replace("\u2019", "'").replace("\x92", "'").replace("\u2018", "'")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def align_meld(sents_by_vid, csv_path, log=print):
    """pkl MELD gốc (không có key_map): ghép từng hội thoại test của pkl với Dialogue_ID trong
    test_sent_emo.csv theo nội dung câu. Trả về {vid: (Dialogue_ID, [Utterance_ID...], score)}.
    Câu thứ i trong pkl <-> câu thứ i (sắp theo Utterance_ID) của hội thoại csv."""
    from difflib import SequenceMatcher
    df = pd.read_csv(csv_path)
    groups = {int(d): g.sort_values("Utterance_ID") for d, g in df.groupby("Dialogue_ID")}
    csv_txt = {d: [_norm_text(x) for x in g["Utterance"]] for d, g in groups.items()}
    by_len = {}
    for d, t in csv_txt.items():
        by_len.setdefault(len(t), []).append(d)

    def score(a, b):
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        idx = sorted(set([0, n // 2, n - 1]))
        return sum(SequenceMatcher(None, a[i], b[i]).ratio() for i in idx) / len(idx)

    out, used, weak = {}, {}, []
    for vid, sents in sents_by_vid.items():
        a = [_norm_text(x) for x in sents]
        cands = by_len.get(len(a)) or list(csv_txt)
        best = max(cands, key=lambda d: score(a, csv_txt[d]))
        sc = score(a, csv_txt[best])
        if sc < 0.6:                                  # cùng độ dài nhưng không giống: thử mọi hội thoại
            best2 = max(csv_txt, key=lambda d: score(a, csv_txt[d]))
            if score(a, csv_txt[best2]) > sc:
                best, sc = best2, score(a, csv_txt[best2])
        if sc < 0.6:
            weak.append((vid, best, round(sc, 2)))
        if best in used:
            log(f"[WARN] hội thoại pkl {vid} và {used[best]} cùng khớp Dialogue_ID {best}")
        used[best] = vid
        out[vid] = (best, groups[best]["Utterance_ID"].astype(int).tolist(), sc)
    n_same = sum(1 for v, (d, _, _) in out.items() if str(v) == str(d))
    log(f"[align] ghép {len(out)} hội thoại pkl với {csv_path.name} theo nội dung câu "
        f"({n_same} có khóa trùng Dialogue_ID, {len(weak)} khớp yếu)")
    if weak:
        log(f"[WARN] khớp yếu (vid, Dialogue_ID, điểm): {weak[:5]}")
    return out


def pkl_test_dialogues(ds, pkl_path, key_map_path=None, csv_path=None, log=print):
    """testVid của pkl -> list dict(rows=DataFrame[key, dialogue, utt, speaker, text, true], t, v, a, slots).
    MELD 14 phần tử: 0 IDs,1 Spk one-hot,2 Lab,4 Txt,8 Aud,9 Vis,10 Sent,12 testVid
    IEMOCAP 12 phần tử: 0 IDs,1 Spk 'M'/'F',2 Lab,3 Txt,7 Aud,8 Vis,9 Sent,11 testVid"""
    data = read_pkl(pkl_path)
    if len(data) == 12:
        IDs, Spk, Lab, Txt, Aud, Vis, Sent, test_vid = (data[0], data[1], data[2], data[3], data[7], data[8],
                                                        data[9], data[11])
    else:
        IDs, Spk, Lab, Txt, Aud, Vis, Sent, test_vid = (data[0], data[1], data[2], data[4], data[8], data[9],
                                                        data[10], data[12])
    if (ds == "iemocap") != (len(data) == 12):
        raise SystemExit(f"pkl {pkl_path} có {len(data)} phần tử, không khớp dataset {ds} "
                         "(IEMOCAP = 12 phần tử, MELD = 14)")
    inv, align = {}, {}
    if ds == "meld" and csv_path and Path(csv_path).is_file():          # cách chắc nhất: ghép theo nội dung
        align = align_meld({v: Sent[v] for v in test_vid}, Path(csv_path), log)
    elif ds == "meld" and key_map_path and Path(key_map_path).is_file():
        km = json.load(open(key_map_path, encoding="utf-8"))["map"]
        inv = {v: int(k.split("/")[1]) for k, v in km.items() if k.startswith("test/")}
    elif ds == "meld":
        log("[WARN] không có test csv lẫn key_map.json: dùng khóa dialogue của pkl làm Dialogue_ID")
    id2 = DIGEMO_ID2LAB[ds]
    out = []
    for vid in test_vid:
        n = len(Lab[vid])
        if ds == "meld":
            if vid in align:
                dia, cu, _ = align[vid]
                utts = [cu[i] if i < len(cu) else 1000 + i for i in range(n)]   # thừa câu -> khóa không khớp
            else:
                dia = int(inv.get(vid, vid))
                utts = [int(u) for u in IDs[vid]]
            keys = [f"dia{dia}_utt{u}" for u in utts]
            spk = [""] * n
            slots = np.asarray(Spk[vid]).argmax(1).tolist()
        else:
            dia = str(vid)
            utts = [str(u) for u in IDs[vid]]
            keys = utts
            spk = [str(s) for s in Spk[vid]]
            slots = [0 if s == "M" else 1 for s in spk]      # như IEMOCAPDataset_BERT: M->[1,0], F->[0,1]
        rows = pd.DataFrame({"key": keys, "dialogue": [dia] * n, "utt": utts, "speaker": spk,
                             "text": [str(s) for s in Sent[vid]],
                             "true": [id2[int(y)] for y in Lab[vid]]})
        out.append({"rows": rows, "t": np.asarray(Txt[vid]), "v": np.asarray(Vis[vid]),
                    "a": np.asarray(Aud[vid]), "slots": slots})
    return out


def pick_device(name):
    import torch
    if name in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def digemo_predict(ds, dialogues, ckpt, device="auto"):
    """chạy checkpoint DiGEmo trên đặc trưng pkl -> (list pred name, list confidence)."""
    for d in ("encoders", "model", "pipeline"):
        p = str(R / "digemo" / d)
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    from predict import load_model_from_checkpoint

    model, cfg, dims, n_classes, ns, dev = load_model_from_checkpoint(str(ckpt), pick_device(device))
    ck_ds = str(getattr(cfg, "dataset", "MELD")).lower()
    if ck_ds != ds:
        raise SystemExit(f"Checkpoint {ckpt} là {ck_ds.upper()}, không phải {ds.upper()}")
    if n_classes != len(DIGEMO_ID2LAB[ds]):
        raise SystemExit(f"Checkpoint có {n_classes} lớp, cần {len(DIGEMO_ID2LAB[ds])}")
    d0 = dialogues[0]
    got = [d0["t"].shape[1], d0["v"].shape[1], d0["a"].shape[1]]
    if list(dims) != got:
        raise SystemExit(f"Kích thước đặc trưng pkl {got} khác checkpoint {list(dims)} (thứ tự t, v, a)")
    print(f"[digemo] {ck_ds.upper()} dims={list(dims)} classes={n_classes} speakers={ns} device={dev}")
    preds, confs = [], []
    with torch.no_grad():
        for d in dialogues:
            L = len(d["slots"])
            t = torch.tensor(d["t"], dtype=torch.float32).unsqueeze(1).to(dev)
            v = torch.tensor(d["v"], dtype=torch.float32).unsqueeze(1).to(dev)
            a = torch.tensor(d["a"], dtype=torch.float32).unsqueeze(1).to(dev)
            q = torch.zeros(L, 1, ns, device=dev)
            q[torch.arange(L), 0, torch.tensor(d["slots"])] = 1.0
            umask = torch.ones(L, 1, device=dev)
            pr = torch.softmax(model(t, v, a, umask, q, [L])[0], -1).float().cpu().numpy()
            preds += [DIGEMO_ID2LAB[ds][i] for i in pr.argmax(1)]
            confs += pr.max(1).round(4).tolist()
    del model
    return preds, confs


def read_digemo_pred(ds, path):
    """predictions.csv cũ của eval_test.py (Dialogue_ID, Utterance_ID, Predicted)."""
    df = pd.read_csv(path)
    if ds == "meld":
        key = [f"dia{int(d)}_utt{int(u)}" for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])]
    else:
        key = df["Utterance_ID"].astype(str).tolist()
    pred = [norm_label(ds, p) for p in df["Predicted"]]
    conf = df["Confidence"].tolist() if "Confidence" in df else [np.nan] * len(df)
    return pd.DataFrame({"key": key, "digemo": pred, "digemo_conf": conf})


# ============================================================================ DM-CER
def read_dmcer_pred(ds, path):
    """file dự đoán DM-CER -> DataFrame[key, dmcer, dmcer_true]."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if "prediction" in df:
        raw = df["prediction"]
    elif "prediction_id" in df:
        raw = df["prediction_id"]
    else:
        raise SystemExit(f"{path}: không có cột 'prediction' / 'prediction_id'")
    id2 = DMCER_ID2LAB[ds]
    if ds == "meld":
        for c in ("Dialogue_ID", "Utterance_ID"):
            if c not in df:
                raise SystemExit(f"{path}: thiếu cột {c}")
        ok = df["Dialogue_ID"].notna() & df["Utterance_ID"].notna()
        df, raw = df[ok], raw[ok]
        key = [f"dia{int(d)}_utt{int(u)}" for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])]
        true_col = "Emotion"
    else:
        if "utt_id" not in df:
            raise SystemExit(f"{path}: thiếu cột utt_id")
        key = df["utt_id"].astype(str).str.strip().tolist()
        true_col = "label"
    out = pd.DataFrame({"key": key, "dmcer": [norm_label(ds, x, id2) for x in raw]})
    out["dmcer_true"] = [norm_label(ds, x) for x in df[true_col]] if true_col in df else None
    bad = out["dmcer"].isna() & pd.Series([str(x) not in ("", "nan") for x in raw]).values
    if bad.any():
        print(f"[WARN] {int(bad.sum())} dự đoán DM-CER không đọc được nhãn, vd: {list(raw[bad.values][:3])}")
    return out.drop_duplicates("key", keep="last")


def run_dmcer_inference(ds, c, device="auto", log=print):
    """gọi inference.py của repo DM-CER (cùng python env) -> ghi c['dmcer_pred']."""
    code = in_root(c["dmcer_code_dir"])
    root = in_root(c["dmcer_data_root"])
    ckpt = in_root(c["dmcer_ckpt"])
    out = in_root(c["dmcer_pred"])
    for name, p in (("dmcer_code_dir", code), ("dmcer_data_root", root), ("dmcer_ckpt", ckpt)):
        if not p or not p.exists():
            raise SystemExit(f"Chạy DM-CER cần {name} tồn tại: {p}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "inference.py", "--data_root", str(root), "--ckpt", str(ckpt),
           "--out_tsv" if ds == "iemocap" else "--out_csv", str(out), "--num_workers", "0"]
    if device not in (None, "", "auto"):
        cmd += ["--device", device]
    log("[dmcer] " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(code), capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-15:])
    if r.returncode != 0:
        raise SystemExit(f"DM-CER inference lỗi (mã {r.returncode}):\n{tail}")
    log(f"[dmcer] xong sau {time.time() - t0:.1f}s -> {out}")
    return tail


# ============================================================================ metrics
def metrics(y, p, labels):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "weighted_f1": float(f1_score(y, p, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
        "per_class": {l: {"f1": float(f1[i]), "precision": float(pr[i]), "recall": float(rc[i]),
                          "support": int(sup[i])} for i, l in enumerate(labels)},
        "confusion": confusion_matrix(y, p, labels=labels).tolist(),
    }


def mcnemar(b, c):
    """b = chỉ DiGEmo đúng, c = chỉ DM-CER đúng; chi2 có hiệu chỉnh liên tục."""
    if b + c == 0:
        return {"chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"chi2": float(chi2), "p": float(math.erfc(math.sqrt(chi2 / 2)))}


def build_table(ds, truth, dig, dm):
    """ghép nhãn thật (pkl) + DiGEmo + DM-CER theo key."""
    t = truth.merge(dig, on="key", how="left").merge(dm, on="key", how="left")
    t["digemo_ok"] = t["digemo"] == t["true"]
    t["dmcer_ok"] = t["dmcer"] == t["true"]
    t["agree"] = t["digemo"] == t["dmcer"]
    both = t["digemo"].notna() & t["dmcer"].notna()
    cat = np.where(~both, "missing",
                   np.where(t["digemo_ok"] & t["dmcer_ok"], "both_correct",
                            np.where(t["digemo_ok"], "only_digemo",
                                     np.where(t["dmcer_ok"], "only_dmcer", "both_wrong"))))
    t["category"] = cat
    return t, both


def compare(ds, truth, dig, dm, extra=None):
    labels = DISPLAY[ds]
    t, both = build_table(ds, truth, dig, dm)
    c = t[both]
    if c.empty:
        raise SystemExit("Không có câu nào có cả hai dự đoán: kiểm tra khóa ghép (Dialogue_ID/Utterance_ID hoặc utt_id)")
    lab_mismatch = 0
    if "dmcer_true" in t and t["dmcer_true"].notna().any():
        m = t["dmcer_true"].notna()
        lab_mismatch = int((t.loc[m, "dmcer_true"] != t.loc[m, "true"]).sum())
    counts = c["category"].value_counts().to_dict()
    b, cc = int(counts.get("only_digemo", 0)), int(counts.get("only_dmcer", 0))
    res = {
        "dataset": ds,
        "labels": labels,
        "label_maps": {"digemo_id2label": DIGEMO_ID2LAB[ds], "dmcer_id2label": DMCER_ID2LAB[ds]},
        "coverage": {
            "truth": int(len(t)),
            "digemo": int(t["digemo"].notna().sum()),
            "dmcer": int(t["dmcer"].notna().sum()),
            "common": int(len(c)),
            "dmcer_extra_keys": int(len(set(dm["key"]) - set(truth["key"]))),
            "label_mismatch_vs_dmcer_file": lab_mismatch,
        },
        "models": {
            "DiGEmo": metrics(c["true"], c["digemo"], labels),
            "DM-CER": metrics(c["true"], c["dmcer"], labels),
        },
        "full_coverage": {
            "DiGEmo": metrics(t.loc[t["digemo"].notna(), "true"], t.loc[t["digemo"].notna(), "digemo"], labels)
            if t["digemo"].notna().any() else None,
            "DM-CER": metrics(t.loc[t["dmcer"].notna(), "true"], t.loc[t["dmcer"].notna(), "dmcer"], labels)
            if t["dmcer"].notna().any() else None,
        },
        "agreement": {
            "both_correct": int(counts.get("both_correct", 0)), "only_digemo": b, "only_dmcer": cc,
            "both_wrong": int(counts.get("both_wrong", 0)),
            "agree_rate": float(c["agree"].mean()),
            "both_wrong_same_label": int(((~c["digemo_ok"]) & (~c["dmcer_ok"]) & c["agree"]).sum()),
            "mcnemar": mcnemar(b, cc),
        },
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        res.update(extra)
    return res, t


# ============================================================================ ghi file
COLS = ["key", "dialogue", "utt", "speaker", "text", "true", "digemo", "dmcer", "digemo_conf",
        "digemo_ok", "dmcer_ok", "agree", "category"]


def summary_md(res):
    L = res["labels"]
    cols = ["Model", "ACC", "W-F1", "Macro-F1"] + [f"{l} F1" for l in L]
    md = f"# {res['dataset'].upper()} test: DiGEmo vs DM-CER ({res['coverage']['common']} câu chung)\n\n"
    md += "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for name, m in res["models"].items():
        vals = [m["accuracy"], m["weighted_f1"], m["macro_f1"]] + [m["per_class"][l]["f1"] for l in L]
        md += f"| {name} | " + " | ".join(f"{100 * v:.2f}" for v in vals) + " |\n"
    a = res["agreement"]
    md += (f"\nCả hai đúng {a['both_correct']} · chỉ DiGEmo đúng {a['only_digemo']} · chỉ DM-CER đúng "
           f"{a['only_dmcer']} · cả hai sai {a['both_wrong']} · trùng dự đoán {100 * a['agree_rate']:.1f}% · "
           f"McNemar p = {a['mcnemar']['p']:.4f}\n")
    cv = res["coverage"]
    md += (f"\nPhủ: pkl {cv['truth']} câu, DiGEmo {cv['digemo']}, DM-CER {cv['dmcer']}, chung {cv['common']}. "
           f"Nhãn thật khác file DM-CER: {cv['label_mismatch_vs_dmcer_file']} câu.\n")
    return md


def plot(res, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:      # không có matplotlib thì vẫn có csv/json
        print("[WARN] bỏ qua vẽ hình:", e)
        return
    L = res["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (name, m) in zip(axes, res["models"].items()):
        cm = np.asarray(m["confusion"], float)
        frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(L)):
            for j in range(len(L)):
                ax.text(j, i, f"{100 * frac[i, j]:.0f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                        color="white" if frac[i, j] > 0.55 else "#172033")
        ax.set_xticks(range(len(L)), L, rotation=45, ha="right")
        ax.set_yticks(range(len(L)), L)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"{name}  ACC {100 * m['accuracy']:.2f}  W-F1 {100 * m['weighted_f1']:.2f}")
    fig.tight_layout()
    fig.savefig(out / "cm_compare.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.1 * len(L) + 3, 4))
    x = np.arange(len(L))
    for k, (name, color) in enumerate((("DiGEmo", "#2563EB"), ("DM-CER", "#EA7A0C"))):
        v = [100 * res["models"][name]["per_class"][l]["f1"] for l in L]
        bars = ax.bar(x + (k - 0.5) * 0.38, v, 0.36, label=name, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(x, L)
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "f1_per_class.png", dpi=200)
    plt.close(fig)


def write_outputs(res, table, out):
    out.mkdir(parents=True, exist_ok=True)
    t = table[[c for c in COLS if c in table]]
    t.to_csv(out / "per_utterance.csv", index=False, encoding="utf-8-sig")
    t[(t["category"] != "missing") & (~t["agree"])].to_csv(out / "disagreements.csv", index=False,
                                                            encoding="utf-8-sig")
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    (out / "summary.md").write_text(summary_md(res), encoding="utf-8")
    plot(res, out)
    full = dict(res)
    full["rows"] = json.loads(t.to_json(orient="records", force_ascii=False))
    json.dump(full, open(out / "compare.json", "w", encoding="utf-8"), ensure_ascii=False)
    return full


# ============================================================================ chạy
def run_compare(ds, cfg, run_dmcer=False, log=print):
    """điểm vào chung cho CLI và web. Trả về dict compare.json."""
    ds = ds.lower()
    if ds not in DATASETS:
        raise SystemExit(f"dataset phải là {DATASETS}")
    c = cfg[ds]
    out = in_root(cfg["out_dir"]) / ds
    pkl = in_root(c["pkl"])
    if not pkl or not pkl.is_file():
        raise SystemExit(f"Không thấy pkl {ds}: {pkl}")
    t0 = time.time()
    dialogues = pkl_test_dialogues(ds, pkl, in_root(c.get("key_map")), in_root(c.get("csv")), log)
    truth = pd.concat([d["rows"] for d in dialogues], ignore_index=True)
    log(f"[truth] {len(truth)} câu / {len(dialogues)} hội thoại test từ {pkl.name}")

    ckpt, dpred = in_root(c.get("digemo_ckpt")), in_root(c.get("digemo_pred"))
    if ckpt and ckpt.is_file():
        p, conf = digemo_predict(ds, dialogues, ckpt, cfg.get("device", "auto"))
        dig = pd.DataFrame({"key": truth["key"], "digemo": p, "digemo_conf": conf})
        dig_src = str(ckpt)
    elif dpred and dpred.is_file():
        dig, dig_src = read_digemo_pred(ds, dpred), str(dpred)
        log(f"[digemo] không có checkpoint, đọc dự đoán có sẵn {dpred}")
    else:
        raise SystemExit(f"Không có checkpoint DiGEmo ({ckpt}) và cũng không có digemo_pred")

    dm_log = ""
    if run_dmcer:
        dm_log = run_dmcer_inference(ds, c, cfg.get("device", "auto"), log)
    dm_path = in_root(c["dmcer_pred"])
    if not dm_path or not dm_path.is_file():
        raise SystemExit(f"Không thấy file dự đoán DM-CER: {dm_path} (sửa dmcer_pred hoặc bật chạy DM-CER)")
    dm = read_dmcer_pred(ds, dm_path)
    log(f"[dmcer] {len(dm)} dự đoán từ {dm_path.name}")

    res, table = compare(ds, truth, dig, dm, extra={
        "sources": {"pkl": str(pkl), "digemo": dig_src, "dmcer": str(dm_path), "dmcer_ran": bool(run_dmcer)},
        "dmcer_log": dm_log, "seconds": round(time.time() - t0, 1)})
    full = write_outputs(res, table, out)
    log(summary_md(res))
    log(f"[ok] -> {out}")
    return full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--write_config", action="store_true", help="ghi file config mẫu rồi thoát")
    ap.add_argument("--run_dmcer", action="store_true", help="gọi inference.py của DM-CER trước khi so sánh")
    ap.add_argument("--out_dir")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    for k in PATH_KEYS:
        ap.add_argument(f"--{k}", help=f"ghi đè <dataset>.{k} trong config")
    a = ap.parse_args()

    if a.write_config:
        p = Path(a.config)
        if p.exists():
            raise SystemExit(f"{p} đã tồn tại, không ghi đè")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(TEMPLATE, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[ok] ghi", p)
        return
    if not a.dataset:
        ap.error("cần --dataset meld|iemocap")
    cfg = load_config(a.config)
    for k in PATH_KEYS:
        if getattr(a, k):
            cfg[a.dataset][k] = getattr(a, k)
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.device:
        cfg["device"] = a.device
    run_compare(a.dataset, cfg, run_dmcer=a.run_dmcer)


if __name__ == "__main__":
    main()
