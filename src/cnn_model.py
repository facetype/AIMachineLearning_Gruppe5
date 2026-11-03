from torch import Tensor, nn

from src.preprocessing import N_LEADS, SUPERCLASSES


def conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> MaxPool. Keeps the length through the conv, then halves it."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class ECGCNN(nn.Module):
    """1D CNN: (batch, N_LEADS, time) -> (batch, len(SUPERCLASSES)) logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(N_LEADS, 32, kernel_size=7),
            conv_block(32, 64, kernel_size=5),
            conv_block(64, 128, kernel_size=5),
            conv_block(128, 128, kernel_size=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, len(SUPERCLASSES))

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
