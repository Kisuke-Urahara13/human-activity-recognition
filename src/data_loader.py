"""Load and validate the official UCI HAR dataset files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

SIGNALS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


@dataclass(frozen=True)
class DatasetSplit:
    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    return path


def load_feature_names(data_dir: str | Path) -> list[str]:
    path = _required(Path(data_dir) / "features.txt")
    names = pd.read_csv(path, sep=r"\s+", header=None)[1].astype(str).tolist()
    # Some feature labels repeat in the source dataset; make DataFrame-safe names.
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        count = seen.get(name, 0)
        unique.append(name if count == 0 else f"{name}_{count}")
        seen[name] = count + 1
    return unique


def load_feature_split(data_dir: str | Path, split: str) -> DatasetSplit:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_dir) / split
    X = np.loadtxt(_required(root / f"X_{split}.txt"), dtype=np.float32)
    y = np.loadtxt(_required(root / f"y_{split}.txt"), dtype=np.int64)
    subjects = np.loadtxt(_required(root / f"subject_{split}.txt"), dtype=np.int64)
    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(f"Inconsistent row counts in {split} split")
    return DatasetSplit(X=X, y=y, subjects=subjects)


def load_raw_split(data_dir: str | Path, split: str) -> DatasetSplit:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(data_dir) / split
    signal_dir = root / "Inertial Signals"
    channels = [
        np.loadtxt(_required(signal_dir / f"{signal}_{split}.txt"), dtype=np.float32)
        for signal in SIGNALS
    ]
    X = np.stack(channels, axis=-1)
    y = np.loadtxt(_required(root / f"y_{split}.txt"), dtype=np.int64)
    subjects = np.loadtxt(_required(root / f"subject_{split}.txt"), dtype=np.int64)
    return DatasetSplit(X=X, y=y, subjects=subjects)

