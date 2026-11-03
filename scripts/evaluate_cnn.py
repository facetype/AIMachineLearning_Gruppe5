"""Evaluate the trained CNN (models/cnn.pt) on the test set and save the metrics to results/.

Run from the repo root:  python -m scripts.evaluate_cnn
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_cnn import BATCH_SIZE, MODEL_PATH
from src.cnn_model import ECGCNN
from src.ecg_dataset import ECGDataset
from src.evaluation import compute_metrics, tune_thresholds
from src.preprocessing import ROOT

RESULTS_PATH = ROOT / "results/cnn_test_metrics.csv"


@torch.no_grad()
def predict_probs(model, dataset, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for every record in dataset."""
    model.eval()
    probs = [torch.sigmoid(model(X.to(device))).cpu()
             for X, _ in DataLoader(dataset, batch_size=BATCH_SIZE)]
    return dataset.y.numpy(), torch.cat(probs).numpy()


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ECGCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    y_val, prob_val = predict_probs(model, ECGDataset("val"), device)
    y_test, prob_test = predict_probs(model, ECGDataset("test"), device)

    # Thresholds are chosen on validation so the test set stays untouched until the final score.
    thresholds = tune_thresholds(y_val, prob_val)

    for name, t in [("threshold 0.5", 0.5), ("thresholds tuned on val", thresholds)]:
        print(f"\nTest set, {name}:")
        print(compute_metrics(y_test, prob_test, t).round(3).to_string())

    compute_metrics(y_test, prob_test, thresholds).to_csv(RESULTS_PATH)
    print(f"\nSaved {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
