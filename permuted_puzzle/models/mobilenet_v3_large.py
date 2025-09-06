from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn

IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD  = [.229, .224, .225]

def build(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2):
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_feats = model.classifier[-1].in_features
    # Replace final classifier (keep MobileNet’s built-in dropout; add optional extra)
    head = []
    if isinstance(model.classifier[0], nn.Sequential):
        # unlikely; safeguard
        pass
    head.append(nn.Linear(in_feats, num_classes))
    if dropout and dropout > 0:
        head.insert(0, nn.Dropout(p=dropout))
    model.classifier[-1] = nn.Sequential(*head)
    meta = {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return model, meta
