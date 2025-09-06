import torch.nn as nn

IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD  = [.229, .224, .225]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        # Input assumed 3x224x224
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # 256x1x1
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build(num_classes: int = 2, pretrained: bool = False, dropout: float = 0.3):
    """
    Simple CNN trained from scratch (no pretrained weights).
    Returns (model, meta) to align with other backbones.
    """
    model = SimpleCNN(num_classes=num_classes, dropout=dropout)
    meta = {"input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return model, meta
