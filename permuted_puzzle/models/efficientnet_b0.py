from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD  = [.229, .224, .225]

def build(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_feats = model.classifier[1].in_features

    if dropout and dropout > 0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),  # index 0
            nn.ReLU(inplace=True),               # index 1
            nn.Linear(in_feats, num_classes)    # index 2
        )
    else:
        # You'll need to update this, too, if you ever use it
        model.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(in_feats, num_classes)
        )

    meta = {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return model, meta