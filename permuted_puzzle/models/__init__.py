from . import resnet18, mobilenet_v3_large, efficientnet_b0, convnext_tiny

REGISTRY = {
    "resnet18": resnet18.build,
    "mobilenet_v3_large": mobilenet_v3_large.build,
    "efficientnet_b0": efficientnet_b0.build,
    "convnext_tiny": convnext_tiny.build,
}
