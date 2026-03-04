"""
models/googlenet.py  —  GoogLeNet (Inception v1) with transfer learning
"""
import torch.nn as nn
from torchvision import models


def get_googlenet(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    
    # If weights are used, torchvision REQUIRES aux_logits=True
    model = models.googlenet(weights=weights, aux_logits=(weights is not None))

    # Disable aux_logits after loading so the model returns a single output during training.
    # We DO NOT delete aux1/aux2 because the internal forward() still checks for them.
    model.aux_logits = False
    
    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
