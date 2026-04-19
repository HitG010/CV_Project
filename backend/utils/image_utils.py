"""
utils/image_utils.py — PIL ↔ tensor ↔ base64 helpers and preprocessing.
"""
import base64, io
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from config import DEVICE

# ── Normalisation constants ────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

# ── Standard transforms ───────────────────────────────────────────
preprocess_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

face_preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

to_tensor = transforms.ToTensor()


def facenet_prewhiten(x: torch.Tensor) -> torch.Tensor:
    """Standardize each image in a batch the way FaceNet expects."""
    if x.ndim != 4:
        raise ValueError("Expected NCHW tensor for FaceNet preprocessing")
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
    std_adj = torch.clamp(std, min=1.0 / (x[0].numel() ** 0.5))
    return (x - mean) / std_adj


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Pixel tensor [0,1] → ImageNet-normalised (in-place safe)."""
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return x * IMAGENET_STD + IMAGENET_MEAN

def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode()


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def tensor_to_b64(tensor: torch.Tensor, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    save_image(tensor, buffer, format=fmt)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()
