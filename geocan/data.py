from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_VIEW_ORDER = ["A4C", "PL", "PSAV", "PSMV", "Random", "SC"]


def seed_everything(seed: int, reproducible: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(reproducible)
    torch.backends.cudnn.benchmark = not bool(reproducible)


def read_image_to_tensor(img_path: str, image_size: int = 224) -> Tensor:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    if img.shape[:2] != (image_size, image_size):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] != 3:
        img = img[:, :, :3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


@dataclass
class SplitConfig:
    split_seed: int = 20
    grade_min: int = 0
    grade_max: int = 10
    view_order: List[str] = None

    def __post_init__(self):
        if self.view_order is None:
            self.view_order = DEFAULT_VIEW_ORDER


def build_dataframe(image_dir: str, grade_dir: str, split_cfg: SplitConfig) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(grade_dir, "*_grades.csv")))
    if not csv_paths:
        raise RuntimeError(f"No *_grades.csv file found in {grade_dir}")

    rows = []
    for csv_path in csv_paths:
        df_i = pd.read_csv(csv_path)
        df_i.columns = [c.strip() for c in df_i.columns]
        df_i = df_i.rename(columns={
            "Image Name": "image_name",
            "ImageName": "image_name",
            "image_name": "image_name",
            "Subfolder": "subfolder",
            "Subfolder Name": "subfolder",
            "SubFolder": "subfolder",
            "subfolder": "subfolder",
            "Grade": "grade",
            "grade": "grade",
        })

        required = {"image_name", "subfolder", "grade"}
        if not required.issubset(df_i.columns):
            raise RuntimeError(f"{csv_path} must contain columns {required}, got {list(df_i.columns)}")

        rows.append(df_i[["image_name", "subfolder", "grade"]])

    df = pd.concat(rows, ignore_index=True)
    df["image_name"] = df["image_name"].astype(str)
    df["subfolder"] = df["subfolder"].astype(str)
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df = df.dropna(subset=["grade"]).copy()
    df["grade"] = df["grade"].astype(int)

    df = df[(df["grade"] >= split_cfg.grade_min) & (df["grade"] <= split_cfg.grade_max)]
    df = df[df["subfolder"].isin(split_cfg.view_order)].copy()

    sub2id: Dict[str, int] = {name: idx for idx, name in enumerate(split_cfg.view_order)}
    df["class_id"] = df["subfolder"].map(sub2id).astype(int)
    df["path"] = df.apply(lambda r: os.path.join(image_dir, r["subfolder"], r["image_name"]), axis=1)
    df = df[df["path"].apply(os.path.isfile)].reset_index(drop=True)

    if "Random" in split_cfg.view_order:
        df.loc[df["subfolder"] == "Random", "grade"] = split_cfg.grade_min

    if len(df) == 0:
        raise RuntimeError("Dataset is empty after filtering paths, labels, and views.")

    return df


def _alloc_4_1_1_counts(n: int):
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 0, 1, 1

    n_test = n // 6
    n_val = n // 6
    n_train = n - n_test - n_val

    if n_test == 0:
        n_test = 1
        n_train -= 1
    if n_val == 0:
        n_val = 1
        n_train -= 1

    while n_train < 0:
        if n_val > 1:
            n_val -= 1
            n_train += 1
        elif n_test > 1:
            n_test -= 1
            n_train += 1
        else:
            break

    return n_train, n_val, n_test


def apply_or_create_fixed_split(df: pd.DataFrame, grade_dir: str, split_cfg: SplitConfig) -> pd.DataFrame:
    cache_path = os.path.join(
        grade_dir,
        f"fixed_split_seed{split_cfg.split_seed}_by_grade_4-1-1.csv",
    )

    if os.path.isfile(cache_path):
        split_df = pd.read_csv(cache_path)
        return df.merge(split_df[["path", "split"]], on="path", how="inner").reset_index(drop=True)

    train_paths, val_paths, test_paths = [], [], []
    for grade in range(split_cfg.grade_min, split_cfg.grade_max + 1):
        paths_g = df[df["grade"] == grade]["path"].tolist()
        if not paths_g:
            continue

        rng = np.random.RandomState(split_cfg.split_seed + grade * 10007)
        idx = np.arange(len(paths_g))
        rng.shuffle(idx)
        paths_g = [paths_g[i] for i in idx]

        n_train, n_val, n_test = _alloc_4_1_1_counts(len(paths_g))
        train_paths.extend(paths_g[:n_train])
        val_paths.extend(paths_g[n_train:n_train + n_val])
        test_paths.extend(paths_g[n_train + n_val:n_train + n_val + n_test])

    records = [(p, "train") for p in train_paths]
    records += [(p, "val") for p in val_paths]
    records += [(p, "test") for p in test_paths]

    split_df = pd.DataFrame(records, columns=["path", "split"]).drop_duplicates("path")
    split_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df.merge(split_df, on="path", how="inner").reset_index(drop=True)


class EchoQualityDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        grade_dir: str,
        split: str,
        split_seed: int = 20,
        image_size: int = 224,
        view_order: List[str] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.image_size = image_size
        self.split_cfg = SplitConfig(split_seed=split_seed, view_order=view_order or DEFAULT_VIEW_ORDER)
        self.sub2id = {name: idx for idx, name in enumerate(self.split_cfg.view_order)}

        df = build_dataframe(image_dir, grade_dir, self.split_cfg)
        df = apply_or_create_fixed_split(df, grade_dir, self.split_cfg)
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise RuntimeError(f"Split '{split}' is empty.")

        self.df = df
        self.paths = df["path"].tolist()
        self.y_class = df["class_id"].astype(int).tolist()
        self.y_score = df["grade"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        x = read_image_to_tensor(self.paths[idx], image_size=self.image_size)
        return x, int(self.y_class[idx]), int(self.y_score[idx]), self.paths[idx]
