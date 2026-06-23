from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from geocan import GeoCAN, GeoCANConfig
from geocan.data import EchoQualityDataset, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Test GeoCAN public release.")
    parser.add_argument("--image-dir", default="Images Dataset")
    parser.add_argument("--grade-dir", default="Grades")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", default="predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--num-score-classes", type=int, default=11)
    return parser.parse_args()


def safe_corr(fn, y_true, y_pred):
    try:
        return float(fn(y_true, y_pred)[0])
    except Exception:
        return float("nan")


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed, reproducible=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = EchoQualityDataset(args.image_dir, args.grade_dir, "test", split_seed=args.split_seed)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    cfg = GeoCANConfig(num_classes=args.num_classes, num_score_classes=args.num_score_classes)
    model = GeoCAN(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []
    for x, y_class, y_score, paths in tqdm(test_loader):
        x = x.to(device, non_blocking=True)
        class_logits, class_pred, score_logits = model(x)

        score_prob = torch.softmax(score_logits, dim=1)
        score_pred = score_logits.argmax(dim=1)
        score_expected = torch.sum(
            score_prob * torch.arange(args.num_score_classes, device=device).float().view(1, -1),
            dim=1,
        )

        for i in range(x.size(0)):
            rows.append({
                "path": paths[i],
                "class_true": int(y_class[i]),
                "class_pred": int(class_pred[i].cpu()),
                "score_true": int(y_score[i]),
                "score_pred": int(score_pred[i].cpu()),
                "score_expected": float(score_expected[i].cpu()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    class_acc = float((df["class_true"] == df["class_pred"]).mean())
    score_acc = float((df["score_true"] == df["score_pred"]).mean())

    y_true = df["score_true"].to_numpy(dtype=float)
    y_expected = df["score_expected"].to_numpy(dtype=float)

    plcc = safe_corr(pearsonr, y_true, y_expected)
    srcc = safe_corr(spearmanr, y_true, y_expected)
    krcc = safe_corr(kendalltau, y_true, y_expected)

    print("Test samples:", len(df))
    print(f"CA:   {class_acc:.4f}")
    print(f"QA:   {score_acc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"KRCC: {krcc:.4f}")

    print("\nClass confusion matrix:")
    print(confusion_matrix(df["class_true"], df["class_pred"]))

    print("\nScore confusion matrix:")
    print(confusion_matrix(df["score_true"], df["score_pred"]))

    print("\nSaved predictions:", args.output_csv)


if __name__ == "__main__":
    main()
