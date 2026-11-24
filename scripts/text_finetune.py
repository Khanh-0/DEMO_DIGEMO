"""Fine-tune RoBERTa-base trên MELD (chỉ câu thoại) + Linear(768->1024) + classifier 7 lớp.
Chọn epoch tốt nhất theo weighted-F1 trên dev, early stopping (patience mặc định 2) để giảm rò rỉ nhãn
sang feature train. Tập test KHÔNG được dùng.

    python text_finetune.py --train_csv train_sent_emo.csv --val_csv dev_sent_emo.csv --out text_ft
Kết quả: text_ft/best.pt
"""
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / 'digemo' / d) for d in ('encoders', 'model', 'pipeline')]
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common import EMO, read_split_csv
from encoders import TextEncoder


def make_loader(df, tok, bs, shuffle, max_len):
    items = list(zip(df["Utterance"].tolist(), [EMO[str(e).lower()] for e in df["Emotion"]]))

    def collate(b):
        enc = tok([t for t, _ in b], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor([y for _, y in b])

    return DataLoader(items, bs, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ps, ys = [], []
    for ids, am, y in loader:
        logits, _ = model(ids.to(dev), am.to(dev))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    return f1_score(y, p, average="weighted"), (p == y).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv",required=True)
    ap.add_argument("--val_csv",required=True)
    ap.add_argument("--out", default="text_ft")
    ap.add_argument("--base", default="roberta-base")
    ap.add_argument("--out_dim", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    tl = make_loader(read_split_csv(args.train_csv), tok, args.bs, True, args.max_len)
    vl = make_loader(read_split_csv(args.val_csv), tok, args.bs * 2, False, args.max_len)

    model = TextEncoder(args.base, args.out_dim).to(dev)
    head = list(model.proj.parameters()) + list(model.fc.parameters())
    opt = torch.optim.AdamW([{"params": model.bert.parameters(), "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    total = args.epochs * len(tl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()

    best, bad = -1.0, 0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for ids, am, y in tl:
            ids, am, y = ids.to(dev), am.to(dev), y.to(dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(ids, am)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        f1, acc = evaluate(model, vl, dev)
        print(f"ep {ep + 1}/{args.epochs}  loss {run / len(tl):.4f}  val_wF1 {f1:.4f}  val_acc {acc:.4f}")
        if f1 > best:
            best, bad = f1, 0
            torch.save({"model": model.state_dict(), "base": args.base, "out_dim": args.out_dim,
                        "val_wf1": f1}, os.path.join(args.out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    print(f"best val_wF1 {best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
