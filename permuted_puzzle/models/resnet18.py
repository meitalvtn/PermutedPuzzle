# permuted_puzzle/models/resnet18.py
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD  = [.229, .224, .225]

def build(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.0):
    """
    Returns (model, meta) where meta has input/normalization expected by the backbone.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_feats = model.fc.in_features
    if dropout and dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feats, num_classes))
    else:
        model.fc = nn.Linear(in_feats, num_classes)

    meta = {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return model, meta
