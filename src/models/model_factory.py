import torch.nn as nn
import torchvision.models as tv_models


def get_model(name: str, num_classes: int, pretrained: bool = False):
    """
    Model factory.

    Supported:
    - resnet18
    - vit_tiny_patch16_224

    For ViT, we use timm because it provides small pretrained ViT models.
    """

    name = name.lower()

    if name == "resnet18":
        model = tv_models.resnet18(weights=None)

        # CIFAR-10 images are small, so i wwisl use a smaller first conv and remove maxpool.
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "vit_tiny_patch16_224":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for ViT models. Install it with: pip install timm"
            ) from exc

        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        return model
    
    if name == "vit_base_patch16_224":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for ViT models. Install it with: pip install timm"
            ) from exc

        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
        )
        return model

    raise ValueError(f"Unknown model name: {name}")