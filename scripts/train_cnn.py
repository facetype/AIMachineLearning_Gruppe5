"""Train the 1D CNN on the processed PTB-XL data and save the best model to models/.

Run from the repo root:  python -m scripts.train_cnn
"""

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cnn_model import ECGCNN
from src.ecg_dataset import ECGDataset
from src.preprocessing import ROOT

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42
MODEL_PATH = ROOT / "models/cnn.pt"
HISTORY_PATH = ROOT / "results/cnn_loss_history.csv"


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()        # clear gradients left over from the previous batch
        loss = loss_fn(model(X), y)  # forward pass + how wrong the predictions are
        loss.backward()              # backpropagation: gradient of the loss for every weight
        optimizer.step()             # gradient descent: nudge every weight against its gradient
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total_loss += loss_fn(model(X), y).item() * len(X)
    return total_loss / len(loader.dataset)


def main() -> None:
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}")

    train_loader = DataLoader(ECGDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset("val"), batch_size=BATCH_SIZE)

    model = ECGCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "saved": is_best})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}"
              + ("  * saved" if is_best else ""))

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    print(f"Best val loss {best_val_loss:.4f}, model saved to {MODEL_PATH.relative_to(ROOT)}")
    print(f"Loss history saved to {HISTORY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
