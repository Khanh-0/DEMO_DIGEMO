"""Giai đoạn 1: train DenseNet-121 (ảnh 128x128) trên FER+.

Mô hình: DenseNet121 features -> GAP(1024) -> Linear(342)+BN+ReLU  (embedding cho DiGEMO)
                                              -> Dropout -> Linear(num_classes)

Cấu trúc dữ liệu (ImageFolder), tải từ Kaggle arnabkumarroy02/ferplus:
    DATA/train/<class>/*.png
    DATA/val/<class>/*.png   (hoặc valid/)
    DATA/test/<class>/*.png  (tùy chọn)

Chạy:
    python densenet_fer.py --data /path/to/ferplus --out ckpt_fer --epochs 30
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

EMB_DIM = 342
IMG_SIZE = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class DenseNetFER(nn.Module):
    def __init__(self, num_classes=8, emb_dim=EMB_DIM, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = models.densenet121(weights=weights).features
        self.proj = nn.Sequential(
            nn.Linear(1024, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(inplace=True)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)

    def embed(self, x):
        f = F.relu(self.features(x), inplace=True)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.proj(f)  # [B, 342]

    def forward(self, x):
        e = self.embed(x)
        return self.fc(self.drop(e)), e


def eval_tf():
    return transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_tf():
    return transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="ckpt_fer")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrain", action="store_true", help="không dùng ImageNet init")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_dir = next((d for d in ("val", "valid", "validation") if os.path.isdir(os.path.join(args.data, d))), None)
    train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), train_tf())
    val_ds = datasets.ImageFolder(os.path.join(args.data, val_dir or "test"), eval_tf())
    classes = train_ds.classes
    print("classes:", classes)

    tl = DataLoader(train_ds, args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = DenseNetFER(len(classes), pretrained=not args.no_pretrain).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        pbar = tqdm(tl, desc=f"ep {ep + 1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits, _ = model(x)
                loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        acc = evaluate(model, vl, device)
        print(f"ep {ep + 1}/{args.epochs}  loss {run / len(tl):.4f}  val_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": classes, "emb_dim": EMB_DIM,
                        "img_size": IMG_SIZE, "val_acc": acc}, os.path.join(args.out, "best.pt"))
    print(f"best val_acc {best:.4f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()