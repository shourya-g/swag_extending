import torch.nn as nn
import torchvision.models as models


def get_model(name: str, num_classes: int):
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model name: {name}")