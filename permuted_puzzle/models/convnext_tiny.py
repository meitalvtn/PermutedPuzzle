from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn

IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD  = [.229, .224, .225]

def build(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.0):
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    in_feats = model.classifier[-1].in_features
    if dropout and dropout > 0:
        model.classifier[-1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feats, num_classes))
    else:
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
    meta = {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return model, meta
