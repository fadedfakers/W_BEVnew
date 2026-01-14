import torch
import numpy as np
from typing import Tuple, Dict, Any

class SynchronizedTransform:
    """
    Base class for data augmentation applied synchronously to 
    Point Clouds, Images, and Bounding Boxes.
    """
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class RandomFlip(SynchronizedTransform):
    """Randomly flips the scene (LIDAR, Image, Mask, Boxes) horizontally."""
    def __init__(self, prob: float = 0.5):
        self.prob = prob

class GlobalRotation(SynchronizedTransform):
    """Applies a random rotation to the point cloud and boxes around the Z-axis."""
    def __init__(self, min_rad: float, max_rad: float):
        self.min_rad = min_rad
        self.max_rad = max_rad

class GlobalScaling(SynchronizedTransform):
    """Applies random scaling to the point cloud and boxes."""
    def __init__(self, min_scale: float, max_scale: float):
        self.min_scale = min_scale
        self.max_scale = max_scale
