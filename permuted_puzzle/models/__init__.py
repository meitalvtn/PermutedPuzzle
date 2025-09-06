from .resnet18 import build as resnet18_build
from .mobilenet_v3_large import build as mobilenet_v3_large_build
from .efficientnet_b0 import build as efficientnet_b0_build
from .convnext_tiny import build as convnext_tiny_build
from .simple_cnn import build as simple_cnn_build

REGISTRY = {
    "resnet18": resnet18_build,
    "mobilenet_v3_large": mobilenet_v3_large_build,
    "efficientnet_b0": efficientnet_b0_build,
    "convnext_tiny": convnext_tiny_build,
    "simple_cnn": simple_cnn_build,
}
