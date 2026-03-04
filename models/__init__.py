"""models/__init__.py"""
from .alexnet import get_alexnet
from .vgg import get_vgg
from .googlenet import get_googlenet
from .resnet import get_resnet
from .efficientnet import get_efficientnet

MODEL_REGISTRY = {
    "alexnet":     get_alexnet,
    "vgg":         get_vgg,
    "googlenet":   get_googlenet,
    "resnet":      get_resnet,
    "efficientnet": get_efficientnet,
}
