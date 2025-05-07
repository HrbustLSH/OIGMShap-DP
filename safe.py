import os
import cv2
import torch
import trimesh
import numpy as np

def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

