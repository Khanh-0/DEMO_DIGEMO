#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tao TSV tap TEST IEMOCAP (Session 5, 31 hoi thoai) cho dmcer_preprocess_multisplit_mtcnn.py --tsv.
Cot: utt_id, start_time, end_time, video_path, label, transcript

- utt_id / label / transcript: lay tu iemocap_multi_features.pkl (testVid = phan tu [11]).
- start_time / end_time: lay tu results/final_best_model_prediction.tsv cua repo (thoi gian tac gia dung);
  thieu thi doc <iemocap_root>/Session5/dialog/EmoEvaluation/<dialog>.txt neu co.
- video_path: <iemocap_root>/Session5/dialog/avi/DivX/<dialog>.avi
Kem theo emotion_map.json (thu tu id ma checkpoint dung).
"""
import os, re, json, pickle, argparse
import pandas as pd

PKL_ID2LAB = {0: "hap", 1: "sad", 2: "neu", 3: "ang", 4: "exc", 5: "fru"}      # thu tu trong pkl
CKPT_LAB2ID = {"neu": 0, "fru": 1, "ang": 2, "sad": 3, "hap": 4, "exc": 5}     # EMOTION2ID cua preprocess/inference

def dialog_of(utt):  # Ses05F_script01_1b_F000 -> Ses05F_script01_1b
    return utt.rsplit("_", 1)[0]

def emo_eval_times(root, dialog):
    sess = "Session" + str(int(dialog[3:5]))
    p = os.path.join(root, sess, "dialog", "EmoEvaluation", dialog + ".txt")
    out = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r"^\[(\d+\.?\d*) - (\d+\.?\d*)\]\s+(\S+)", line)
                if m:
                    out[m.group(3)] = (float(m.group(1)), float(m.group(2)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--ref_tsv", default="", help="(tuy chon) code/IEMOCAP/results/final_best_model_prediction.tsv; bo trong -> lay thoi gian tu EmoEvaluation")
    ap.add_argument("--iemocap_root", required=True, help=r"Thu muc IEMOCAP_full_release, vd E:\...\IEMOCAP_full_release")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    d = pickle.load(open(a.pkl, "rb"), encoding="latin1")
    ids, labs, sents, test_dias = d[0], d[2], d[9], d[11]
    ref = pd.read_csv(a.ref_tsv, sep="\t").set_index("utt_id") if a.ref_tsv else pd.DataFrame(columns=["start_time", "end_time"])

    rows, emo_cache, skipped = [], {}, []
    for dia in sorted(test_dias):
        sess = "Session" + str(int(dia[3:5]))
        vpath = os.path.join(a.iemocap_root, sess, "dialog", "avi", "DivX", dia + ".avi")
        for u, l, s in zip(ids[dia], labs[dia], sents[dia]):
            if u in ref.index:
                st, en = float(ref.at[u, "start_time"]), float(ref.at[u, "end_time"])
            else:
                if dia not in emo_cache:
                    emo_cache[dia] = emo_eval_times(a.iemocap_root, dia)
                if u not in emo_cache[dia]:
                    skipped.append(u); continue
                st, en = emo_cache[dia][u]
            rows.append({"utt_id": u, "start_time": st, "end_time": en, "video_path": vpath,
                         "label": PKL_ID2LAB[int(l)], "transcript": str(s).replace("\t", " ").replace("\n", " ")})

    os.makedirs(a.out_dir, exist_ok=True)
    out_tsv = os.path.join(a.out_dir, "iemocap_test.tsv")
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    with open(os.path.join(a.out_dir, "emotion_map.json"), "w") as f:
        json.dump(CKPT_LAB2ID, f, indent=2)

    missing_v = sorted({r["video_path"] for r in rows if not os.path.isfile(r["video_path"])})
    print(f"Ghi {len(rows)} cau / {len(test_dias)} hoi thoai -> {out_tsv}")
    if skipped: print(f"Bo qua {len(skipped)} cau (khong co thoi gian), vd: {skipped[:5]}")
    if missing_v: print(f"CANH BAO: {len(missing_v)} video khong ton tai, vd: {missing_v[0]}")

if __name__ == "__main__":
    main()
