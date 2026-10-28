import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.logistic_regression_model import ECGLogisticRegression
from src.ecg_dataset import ECGDataset
from src.preprocessing import ROOT, SUPERCLASSES

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 
SEED = 42

MODEL_PATH = ROOT / "models/logistic_regression.pt"
HISTORY_PATH = ROOT / "results/logistic_loss_history.csv"

# Creates the destination folders if they don't exist 
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for X, y in loader:

        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)

        # As mentioned earlier in the "logistic_regression_model.py" file, it only uses the nn.Linear and miss the Sigmoid function.
        # Here the loss function "nn.BCEWithLogitsLoss" use the sigmoid function first on the outputs before calculating the error against our true labels 'y'
        
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data splitter")

    train_loader = DataLoader(ECGDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset("val"), batch_size=BATCH_SIZE, shuffle=False)

   
    model = ECGLogisticRegression(num_classes=len(SUPERCLASSES)).to(device)
    loss_fn = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    history = []

    print(f"\nBegin training on {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} Training loss {train_loss:.4f} | Validation loss: {val_loss:.4f}")
        
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # 6. Saving the loss-historic in a CSV-fil on HISTORY_PATH, like it has been done in the (train_cnn.py)
    df_history = pd.DataFrame(history)
    df_history.to_csv(HISTORY_PATH, index=False)
    print(f"Loss-historic saved to: {HISTORY_PATH}")


if __name__ == "__main__":
    main()


