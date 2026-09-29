import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sve import SVE

# name: (class, num_classes, train kwargs, val kwargs or None, test kwargs)
DATASETS = {
    "flowers102": (datasets.Flowers102, 102, {"split": "train"}, {"split": "val"}, {"split": "test"}),
    "dtd": (datasets.DTD, 47, {"split": "train"}, {"split": "val"}, {"split": "test"}),
    "aircraft": (datasets.FGVCAircraft, 100, {"split": "train"}, {"split": "val"}, {"split": "test"}),
    "pets": (datasets.OxfordIIITPet, 37, {"split": "trainval"}, None, {"split": "test"}),
    "food101": (datasets.Food101, 101, {"split": "train"}, None, {"split": "test"}),
    "cifar100": (datasets.CIFAR100, 100, {"train": True}, None, {"train": False}),
}


def parse_args():
    p = argparse.ArgumentParser("SVE: Singular Value Ensemble on DINO")
    p.add_argument("--dataset", default="flowers102", choices=list(DATASETS))
    p.add_argument("--data_root", default="./data")
    p.add_argument("--backbone", default="vit_small_patch16_224.dino")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--n_members", type=int, default=4)
    p.add_argument("--init_std", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default="checkpoints")
    return p.parse_args()


def get_loaders(args):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.image_size * 256 / 224)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), norm,
    ])
    cls, num_classes, train_kw, val_kw, test_kw = DATASETS[args.dataset]
    make = lambda kw, tf: cls(root=args.data_root, download=True, transform=tf, **kw)

    train_ds, test_ds = make(train_kw, train_tf), make(test_kw, eval_tf)
    if val_kw is not None:
        val_ds = make(val_kw, eval_tf)
    else:  # hold out part of the training set for validation
        idx = list(range(len(train_ds)))
        random.Random(args.seed).shuffle(idx)
        n_val = int(len(idx) * args.val_fraction)
        val_ds = Subset(make(train_kw, eval_tf), idx[:n_val])
        train_ds = Subset(train_ds, idx[n_val:])

    loader = lambda ds, shuffle: DataLoader(ds, args.batch_size, shuffle=shuffle,
                                            num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False), num_classes


def train_one_epoch(model, loader, optimizer, scaler, args):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.autocast(args.device, enabled=args.amp):
            logits = model(x)  # [B, M, C]
            # every member is trained on the full batch with its own CE loss
            loss = F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(args.n_members))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach().item() * len(y)
        correct += logits.mean(1).argmax(1).eq(y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    """Ensemble prediction = mean of member logits."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().mean(1).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def ece_score(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return float(ece)


def metrics(logits, labels, T=1.0):
    probs = F.softmax(logits / T, dim=1)
    onehot = F.one_hot(labels, probs.shape[1]).float()
    return {
        "acc": probs.argmax(1).eq(labels).float().mean().item(),
        "nll": F.cross_entropy(logits / T, labels).item(),
        "brier": ((probs - onehot) ** 2).sum(1).mean().item(),
        "ece": ece_score(probs, labels),
    }


def fit_temperature(logits, labels):
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return T.item()


def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, num_classes = get_loaders(args)
    model = SVE(args.backbone, num_classes, args.n_members, args.init_std).to(args.device)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):  # linear warmup + cosine decay
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, f"sve_{args.dataset}_M{args.n_members}.pth")
    best = -1.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, args)
        scheduler.step()
        val_acc = metrics(*predict(model, val_loader, args.device))["acc"]
        print(f"Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=args.device))
    val_logits, val_labels = predict(model, val_loader, args.device)
    test_logits, test_labels = predict(model, test_loader, args.device)
    T = fit_temperature(val_logits, val_labels)

    print(f"\nBest val acc: {best:.4f}")
    for name, t in [("Test", 1.0), (f"Test (T={T:.3f})", T)]:
        m = metrics(test_logits, test_labels, t)
        print(f"{name}: acc={m['acc']:.4f} nll={m['nll']:.4f} brier={m['brier']:.4f} ece={m['ece']:.4f}")


if __name__ == "__main__":
    main()
