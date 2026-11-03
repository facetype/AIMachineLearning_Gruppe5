"""Load, clean and split the PTB-XL 100 Hz recordings into model-ready arrays."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
PROCESSED_DIR = ROOT / "data/processed"

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLING_RATE = 100
N_SAMPLES = 10 * SAMPLING_RATE
N_LEADS = 12

# Split recommended by the PTB-XL authors: folds 9 and 10 have human-validated labels.
VAL_FOLD = 9
TEST_FOLD = 10

# Metadata columns with more missing values than this are too sparse to be useful.
MAX_MISSING_FRACTION = 0.6


def load_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read ptbxl_database.csv and add a `superclasses` column (list of SUPERCLASSES)."""
    df = pd.read_csv(raw_dir / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(raw_dir / "scp_statements.csv", index_col=0)
    code_to_class = scp.loc[scp.diagnostic == 1, "diagnostic_class"].to_dict()

    # Every listed code counts, regardless of its likelihood value, as in the PTB-XL benchmark.
    codes = df.scp_codes.apply(ast.literal_eval)
    df["superclasses"] = codes.apply(
        lambda d: sorted({code_to_class[c] for c in d if c in code_to_class})
    )
    return df


def drop_sparse_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop metadata columns missing more than MAX_MISSING_FRACTION of their values."""
    missing = df.isna().mean()
    dropped = missing[missing > MAX_MISSING_FRACTION].index.tolist()
    return df.drop(columns=dropped), dropped


def load_signals(df: pd.DataFrame, raw_dir: Path = RAW_DIR) -> np.ndarray:
    """Read every record in df, returning shape (n_records, N_SAMPLES, N_LEADS) in mV."""
    X = np.empty((len(df), N_SAMPLES, N_LEADS), dtype=np.float32)
    for i, filename in enumerate(df.filename_lr):
        signal, _ = wfdb.rdsamp(str(raw_dir / filename))
        if signal.shape != (N_SAMPLES, N_LEADS):
            raise ValueError(f"{filename}: expected {(N_SAMPLES, N_LEADS)}, got {signal.shape}")
        X[i] = signal
    return X


def find_bad_records(df: pd.DataFrame, X: np.ndarray) -> dict[str, pd.Index]:
    """Return ecg_ids to drop, grouped by reason."""
    no_label = df.superclasses.str.len() == 0
    non_finite = ~np.isfinite(X).all(axis=(1, 2))
    flat_lead = (X.std(axis=1) == 0).any(axis=1)
    return {
        "no diagnostic superclass": df.index[no_label],
        "NaN/inf in signal": df.index[non_finite],
        "flat (constant) lead": df.index[flat_lead],
    }


def to_multi_hot(superclasses: pd.Series) -> np.ndarray:
    y = np.zeros((len(superclasses), len(SUPERCLASSES)), dtype=np.float32)
    for i, classes in enumerate(superclasses):
        for c in classes:
            y[i, SUPERCLASSES.index(c)] = 1
    return y


def split_folds(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks per split, based on strat_fold."""
    fold = df.strat_fold.to_numpy()
    return {
        "train": (fold != VAL_FOLD) & (fold != TEST_FOLD),
        "val": fold == VAL_FOLD,
        "test": fold == TEST_FOLD,
    }


def save_split(name: str, X: np.ndarray, y: np.ndarray, ecg_ids: np.ndarray,
               mean: np.ndarray, std: np.ndarray, out_dir: Path = PROCESSED_DIR) -> Path:
    path = out_dir / f"ptbxl{SAMPLING_RATE}_{name}.npz"
    np.savez(path, X=X, y=y, ecg_id=ecg_ids, mean=mean, std=std,
             classes=np.array(SUPERCLASSES))
    return path


def load_split(name: str, out_dir: Path = PROCESSED_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a processed split: X (n, N_SAMPLES, N_LEADS), y (n, len(SUPERCLASSES)), ecg_id (n,)."""
    with np.load(out_dir / f"ptbxl{SAMPLING_RATE}_{name}.npz") as data:
        return data["X"], data["y"], data["ecg_id"]
