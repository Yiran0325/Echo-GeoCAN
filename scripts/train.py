from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from geocan import GeoCAN, GeoCANConfig
from geocan.data import EchoQualityDataset, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoCAN public release.")
    parser.add_argument("--image-dir", default="Images Dataset")
    parser.add_argument("--grade-dir", default="Grades")
    parser.add_argument("--save-path", default="model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--w-class", type=float, default=0.1)
    parser.add_argument("--w-score", type=float, default=2.0)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--num-score-classes", type=int, default=11)
    parser.add_argument("--use-geo-loss", action="store_true",
                        help="Requires the restricted graph-BPR ranking implementation.")
    return parser.parse_args()


@torch.no_grad()
def acc_from_logits(logits, target):
    return (logits.argmax(dim=1) == target).float().mean().item()


def run_one_epoch(model, loader, device, criterion_class, criterion_score,
                  optimizer=None, w_class=0.1, w_score=2.0, use_geo_loss=False, geo_weight=0.1):
    train = optimizer is not None
    model.train(train)

    total_loss, total_class_acc, total_score_acc, total_geo, n = 0.0, 0.0, 0.0, 0.0, 0

    iterator = tqdm(loader, leave=False)
    for x, y_class, y_score, _paths in iterator:
        x = x.to(device, non_blocking=True)
        y_class = y_class.to(device, non_blocking=True)
        y_score = y_score.to(device, non_blocking=True)

        if use_geo_loss:
            class_logits, _, score_logits, loss_geo = model(x, return_geo_loss=True)
        else:
            class_logits, _, score_logits = model(x, return_geo_loss=False)
            loss_geo = None

        loss_class = criterion_class(class_logits, y_class)
        loss_score = criterion_score(score_logits, y_score)
        loss = w_class * loss_class + w_score * loss_score

        if loss_geo is not None:
            loss = loss + geo_weight * loss_geo

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_class_acc += acc_from_logits(class_logits, y_class) * bs
        total_score_acc += acc_from_logits(score_logits, y_score) * bs
        total_geo += (0.0 if loss_geo is None else float(loss_geo.detach().cpu())) * bs
        n += bs

        iterator.set_description(f"loss={total_loss / max(n,1):.4f}")

    return total_loss / n, total_class_acc / n, total_score_acc / n, total_geo / n


def main():
    args = parse_args()
    seed_everything(args.seed, reproducible=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = EchoQualityDataset(args.image_dir, args.grade_dir, "train", split_seed=args.split_seed)
    val_set = EchoQualityDataset(args.image_dir, args.grade_dir, "val", split_seed=args.split_seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cfg = GeoCANConfig(num_classes=args.num_classes, num_score_classes=args.num_score_classes)
    model = GeoCAN(cfg).to(device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_score = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("train/val:", len(train_set), len(val_set))
    print("view mapping:", train_set.sub2id)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ca, train_qa, train_geo = run_one_epoch(
            model, train_loader, device, criterion_class, criterion_score,
            optimizer=optimizer,
            w_class=args.w_class,
            w_score=args.w_score,
            use_geo_loss=args.use_geo_loss,
            geo_weight=cfg.geo_weight,
        )
        val_loss, val_ca, val_qa, val_geo = run_one_epoch(
            model, val_loader, device, criterion_class, criterion_score,
            optimizer=None,
            w_class=args.w_class,
            w_score=args.w_score,
            use_geo_loss=args.use_geo_loss,
            geo_weight=cfg.geo_weight,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, CA {train_ca:.4f}, QA {train_qa:.4f}, geo {train_geo:.4f} | "
            f"val loss {val_loss:.4f}, CA {val_ca:.4f}, QA {val_qa:.4f}, geo {val_geo:.4f}"
        )

    torch.save(model.state_dict(), args.save_path)
    print("Saved:", args.save_path)


if __name__ == "__main__":
    main()
