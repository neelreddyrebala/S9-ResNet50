# src/train_tinyimagenet.py
import argparse, time, json
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment
from timm.data import Mixup
import timm
from torch.cuda.amp import GradScaler, autocast


def get_loaders(data_root, img_size=224, bs=128, workers=8):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # AutoAugment policy tuned for ImageNet-like data
        transforms.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # RandomErasing ~ Cutout
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(Path(data_root) / "train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(Path(data_root) / "val",   transform=val_tfms)

    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=workers,
        pin_memory=True
    )
    return train_dl, val_dl, len(train_ds.classes)


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval().to(device)
    ce = nn.CrossEntropyLoss()
    n = 0
    loss_sum = 0.0
    top1 = 0
    top5 = 0
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)

        _, pred = logits.topk(5, 1, True, True)
        correct = pred.eq(y.view(-1, 1).expand_as(pred))
        top1 += correct[:, :1].sum().item()
        top5 += correct.sum().item()
        n += x.size(0)

    return {
        "loss": loss_sum / max(1, n),
        "top1": 100.0 * top1 / max(1, n),
        "top5": 100.0 * top5 / max(1, n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path with train/ and val/ dirs")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save", default="runs/tiny-r50")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, num_classes = get_loaders(
        args.data_root, args.img_size, args.batch_size, args.workers
    )

    # FROM SCRATCH: pretrained=False
    model = timm.create_model("resnet50", pretrained=False, num_classes=num_classes).to(device)

    # Optim, sched, AMP
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    # cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.fp16)

    # Mixup/CutMix + label smoothing
    mixup_fn = Mixup(
        mixup_alpha=args.mixup if args.mixup > 0 else 0.0,
        cutmix_alpha=args.cutmix if args.cutmix > 0 else 0.0,
        label_smoothing=args.label_smoothing,
        num_classes=num_classes,
    ) if (args.mixup > 0 or args.cutmix > 0 or args.label_smoothing > 0) else None

    # If using Mixup/CutMix, use soft-target loss
    if mixup_fn is not None:
        ce = timm.loss.SoftTargetCrossEntropy()
    else:
        ce = nn.CrossEntropyLoss()

    out = Path(args.save)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        t0 = time.time()

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mixup_fn is not None:
                x, y = mixup_fn(x, y)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.fp16):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)

        scheduler.step()

        train_loss = epoch_loss / max(1, n)
        val_metrics = evaluate(model, val_dl, device)
        t1 = time.time()

        rec = {
            "epoch": epoch,
            "train": {"loss": train_loss},
            "val": val_metrics,
            "time_sec": round(t1 - t0, 2),
            "pretrained": False,
            "aug": {
                "autoaugment": True, "colorjitter": True, "random_erasing": True,
                "mixup": args.mixup, "cutmix": args.cutmix, "label_smoothing": args.label_smoothing
            },
            "sched": "cosine",
        }
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec)

        torch.save(model.state_dict(), out / f"epoch{epoch:03d}.pt")

    print("Done.")


if __name__ == "__main__":
    main()
