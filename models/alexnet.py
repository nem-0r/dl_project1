"""
models/alexnet.py  —  AlexNet with transfer learning
"""
import torch.nn as nn
from torchvision import models


def get_alexnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)

    # Replace the classifier final layer
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model
