"""Giai đoạn 2: dùng checkpoint FER+ để trích embedding 342-d cho từng utterance.

Đầu vào (chọn 1):
  --json  file json dạng {"dia0_utt2": ["test/dia0_utt2/4.jpg", ...], ...}
  --root  thư mục chứa các folder diaX_uttY/*.jpg (quét tự động, có thể lồng split train/dev/test)

Đầu ra (--out, .pkl): dict  key -> np.ndarray
  --pool mean : [342]            (mean các frame của utterance)  [mặc định]
  --pool none : [T, 342]         (giữ từng frame, theo thứ tự số)

Chạy:
    python extract_face_embed.py --ckpt ckpt_fer/best.pt --json faces.json --out face_342.pkl
"""
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'digemo' / d) for d in ('encoders', 'model', 'pipeline')]
import argparse
import glob
import json
import os
import pickle
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from densenet_fer import DenseNetFER, eval_tf


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]


def collect_from_root(root):
    groups = {}
    for d in glob.glob(os.path.join(root, "**", "dia*_utt*"), recursive=True):
        if os.path.isdir(d):
            files = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))
            if files:
                groups[os.path.basename(d)] = sorted(files, key=natural_key)
    return groups


class FrameDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf  # items: [(key, idx, path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, j, p = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json")
    ap.add_argument("--root")
    ap.add_argument("--out", default="face_342.pkl")
    ap.add_argument("--pool", choices=["mean", "none"], default="mean")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default="", help="thư mục gốc ghép với path tương đối trong json (mặc định: thư mục chứa file json)")
    args = ap.parse_args()
    assert args.json or args.root, "cần --json hoặc --root"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    model = DenseNetFER(len(ck["classes"]), emb_dim=ck["emb_dim"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            groups = json.load(f)
        base = args.base or os.path.dirname(os.path.abspath(args.json))
        groups = {k: sorted([os.path.join(base, p) for p in v], key=natural_key) for k, v in groups.items()}
    else:
        groups = collect_from_root(args.root)

    items = [(k, j, p) for k, ps in groups.items() for j, p in enumerate(ps)]
    print(f"{len(groups)} utterances, {len(items)} frames")
    loader = DataLoader(FrameDS(items, eval_tf()), args.bs, shuffle=False, num_workers=args.workers)

    emb = np.zeros((len(items), ck["emb_dim"]), dtype=np.float32)
    with torch.no_grad():
        for x, idx in tqdm(loader, desc="extract"):
            emb[idx.numpy()] = model.embed(x.to(device)).float().cpu().numpy()

    out, pos = {}, 0
    for k, ps in groups.items():
        e = emb[pos:pos + len(ps)]
        pos += len(ps)
        out[k] = e.mean(0) if args.pool == "mean" else e

    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"saved {len(out)} entries -> {args.out}")


if __name__ == "__main__":
    main()