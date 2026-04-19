"""
config.py — Mirage-AI v2
Centralised device selection and shared paths.
"""
import os
import torch
from pathlib import Path

# ── Device ────────────────────────────────────────────────────────
DEVICE: str = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGENET_CLASSES_FILE = DATA_DIR / "imagenet_classes.txt"
CIFAR10_ROOT = os.getenv("CIFAR10_ROOT", str(DATA_DIR / "cifar10"))

# ── ImageNet class list ───────────────────────────────────────────
def load_imagenet_classes() -> list[str]:
    if not IMAGENET_CLASSES_FILE.exists():
        raise FileNotFoundError(
            f"imagenet_classes.txt not found at {IMAGENET_CLASSES_FILE}.\n"
            "Download it from: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt\n"
            f"and place it at: {IMAGENET_CLASSES_FILE}"
        )
    with open(IMAGENET_CLASSES_FILE) as class_file:
        return [line.strip() for line in class_file]

IDX_TO_CLASS: list[str] = load_imagenet_classes()
