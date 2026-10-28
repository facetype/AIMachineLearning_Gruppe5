# Support vector machine here!
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation import compute_metrics
from src.preprocessing import ROOT, load_split

RESULTS_PATH = ROOT / "results/svm_test_metrics.csv"


def make_features(X: np.ndarray) -> np.ndarray:
    """Six statistical features per lead -> 72 features per ECG."""
    return np.concatenate([
        np.mean(X, axis=1),
        np.std(X, axis=1),
        np.min(X, axis=1),
        np.max(X, axis=1),
        np.percentile(X, 25, axis=1),
        np.percentile(X, 75, axis=1),
    ], axis=1)


def main() -> None:
    # Load data
    X_train, y_train, _ = load_split("train")
    X_test, y_test, _ = load_split("test")

    # Feature extraction
    features_train = make_features(X_train)
    features_test = make_features(X_test)

    # Standardization
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # One SVM per superclass, RBF kernel, balanced class weights
    svm = OneVsRestClassifier(SVC(kernel="rbf", class_weight="balanced"), n_jobs=-1)
    svm.fit(features_train, y_train)

    # Evaluation: decision scores are positive for "yes", so the threshold is 0
    scores = svm.decision_function(features_test)
    metrics = compute_metrics(y_test, scores, thresholds=0.0)
    print(metrics.round(3).to_string())
    metrics.to_csv(RESULTS_PATH)
    print(f"Saved {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
