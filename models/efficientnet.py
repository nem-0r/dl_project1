"""
models/efficientnet.py  —  EfficientNet-B4 with transfer learning
Modern (2019), state-of-the-art accuracy, efficient architecture.
"""
import torch.nn as nn
from torchvision import models


def get_efficientnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b4(weights=weights)

    # Replace the classifier final layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
