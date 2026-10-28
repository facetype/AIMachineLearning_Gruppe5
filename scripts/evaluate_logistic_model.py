import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_logistic_model import BATCH_SIZE, MODEL_PATH
from src.logistic_regression_model import ECGLogisticRegression
from src.ecg_dataset import ECGDataset
from src.evaluation import compute_metrics, tune_thresholds
from src.preprocessing import ROOT

# Path to save the logstic regression test metrics
RESULTS_PATH = ROOT / "results/logsitic_test_metrics.csv"

@torch.no_grad()
def predict_probs(model, dataset, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    # Here we apply torch.sigmoid() to transform the raw output numbers into true probability between 0.0 and 1.0 so that the metrics can be evaluated correctly.

    probs = [torch.sigmoid(model(X.to(device))).cpu()
             for X, _ in DataLoader(dataset, batch_size=BATCH_SIZE)]
    return dataset.y.numpy(), torch.cat(probs).numpy()

def main() -> None:
    device ="mps" if torch.backends.mps.is_available() else "cpu"

    model = ECGLogisticRegression().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location = device))

    y_val, prob_val = predict_probs(model, ECGDataset("val"), device)
    y_test, prob_test = predict_probs(model, ECGDataset("test"), device)

    thresholds = tune_thresholds(y_val, prob_val)

    for name, t in [("threshold 0.5", 0.5), ("thresholds tuned on val", thresholds)]:
        print(f"\nTest set, {name}:")
        print(compute_metrics(y_test, prob_test,t).round(3).to_string())

    compute_metrics(y_test, prob_test, thresholds).to_csv(RESULTS_PATH)
    print(f"\nSaved {RESULTS_PATH.relative_to(ROOT)}")

if __name__ == "__main__":
    main()

