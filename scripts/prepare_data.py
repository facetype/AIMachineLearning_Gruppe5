"""Build data/processed/ptbxl100_{train,val,test}.npz from the raw PTB-XL 100 Hz records.

Run from the repo root:  python -m scripts.prepare_data
"""

import numpy as np

from src.preprocessing import (
    SUPERCLASSES,
    find_bad_records,
    load_metadata,
    load_signals,
    save_split,
    split_folds,
    to_multi_hot,
)


def main() -> None:
    df = load_metadata()
    print(f"Loading {len(df)} records...")
    X = load_signals(df)

    bad = find_bad_records(df, X)
    drop_ids = set()
    for reason, ids in bad.items():
        print(f"  dropping {len(ids):4d}: {reason}" + (f" {list(ids)}" if 0 < len(ids) <= 5 else ""))
        drop_ids.update(ids)
    keep = ~df.index.isin(list(drop_ids))
    df, X = df[keep], X[keep]
    print(f"Kept {len(df)} records")

    y = to_multi_hot(df.superclasses)
    masks = split_folds(df)

    # Per-lead standardization, fitted on train only so val/test stay unseen.
    mean = X[masks["train"]].mean(axis=(0, 1), dtype=np.float64)
    std = X[masks["train"]].std(axis=(0, 1), dtype=np.float64)
    X = ((X - mean) / std).astype(np.float32)

    ecg_ids = df.index.to_numpy()
    for name, mask in masks.items():
        path = save_split(name, X[mask], y[mask], ecg_ids[mask], mean, std)
        counts = ", ".join(f"{c} {int(n)}" for c, n in zip(SUPERCLASSES, y[mask].sum(axis=0)))
        print(f"{name:5s}: {mask.sum():5d} records ({counts}) -> {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
