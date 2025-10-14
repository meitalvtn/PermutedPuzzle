from torchvision.models import swin_t, Swin_T_Weights
import torch.nn as nn

IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD  = [.229, .224, .225]

def build(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.0):
    """
    Build Swin Transformer Tiny model.

    Architecture details:
        - Input size: 224x224
        - Patch size: 4x4
        - Window size: 7x7
        - Number of heads: [3, 6, 12, 24]
        - Embed dim: 96
        - Depths: [2, 2, 6, 2]

    Notable layers for GradCAM:
        - permute: Final feature map (7x7 spatial resolution) - recommended
        - features.5.1.norm2: Stage 3, last block (14x14 spatial resolution)
        - features.7.1.norm2: Stage 4, last block (7x7 spatial resolution)

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout rate for the classifier head

    Returns:
        (model, meta) tuple where meta contains input_size, mean, std
    """
    weights = Swin_T_Weights.DEFAULT if pretrained else None
    model = swin_t(weights=weights)

    in_feats = model.head.in_features
    if dropout and dropout > 0:
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, num_classes)
        )
    else:
        model.head = nn.Linear(in_feats, num_classes)

    meta = {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return model, meta
