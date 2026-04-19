"""
models/face_models.py — Face detection + embedding models.
"""
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from config import DEVICE

# MTCNN must stay on CPU — MPS/CUDA don't support all its ops.
MTCNN_DET = MTCNN(keep_all=False, device="cpu")

# Face embedding model used by the cloak attacks.
FACENET = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
