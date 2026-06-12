import timm
import torch
from torch import Tensor, nn


class Backbone(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim: int = self.model.num_features

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
