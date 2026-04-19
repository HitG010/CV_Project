"""
models/art_models.py — ImageNet classification ensemble.
"""
import torch
import torchvision.models as models
from config import DEVICE


def load_art_models() -> dict[str, torch.nn.Module]:
    """Load ResNet50, VGG16, DenseNet121 (ImageNet pretrained, eval mode)."""
    resnet50_model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    ).eval().to(DEVICE)
    vgg16_model = models.vgg16(
        weights=models.VGG16_Weights.IMAGENET1K_V1
    ).eval().to(DEVICE)
    densenet_model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
    ).eval().to(DEVICE)
    return {
        "resnet50": resnet50_model,
        "vgg16": vgg16_model,
        "densenet121": densenet_model,
    }


# Module-level singletons (imported by attacks & routes)
ART_MODELS: dict[str, torch.nn.Module] = load_art_models()
PRIMARY_MODEL: torch.nn.Module = ART_MODELS["resnet50"]

# Ensemble weights (resnet50, vgg16, densenet121)
ENSEMBLE_WEIGHTS = (0.5, 0.3, 0.2)


def ensemble_logits(
    x_norm: torch.Tensor,
    weights: tuple = ENSEMBLE_WEIGHTS,
) -> torch.Tensor:
    """Weighted average of logits across the classification ensemble."""
    return sum(
        w * mdl(x_norm)
        for (_, mdl), w in zip(ART_MODELS.items(), weights)
    )
