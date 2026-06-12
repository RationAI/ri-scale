from torch import Tensor, nn


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).squeeze(1)  # (B,)
