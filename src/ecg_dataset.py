import torch
from torch.utils.data import Dataset

from src.preprocessing import load_split


class ECGDataset(Dataset):
    def __init__(self, split: str):
        X, y, _ = load_split(split)
        # Conv1d expects (records, leads, time); the processed files are (records, time, leads).
        self.X = torch.from_numpy(X.transpose(0, 2, 1))
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # signal: (12, 1000), label: (5,)
        return self.X[i], self.y[i]
