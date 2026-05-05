import cv2
import numpy as np
import random
from abc import ABC, abstractmethod


class Alteration(ABC):
    """
    Abstract base class for all image alterations.
    Demonstrates INHERITANCE and POLYMORPHISM.
    """

    def __init__(self, region_size: int = 50):
        self.region_size = region_size   # encapsulation

    @abstractmethod
    def apply(self, image, region):
        """
        Apply alteration to a DifferenceRegion.
        Must be implemented by subclasses.
        """
        pass


class ColorShiftAlteration(Alteration):
    """
    Slightly shifts colours in the region.
    """

    def apply(self, image, region):
        x, y, size = region.x, region.y, region.size
        roi = image[y:y+size, x:x+size]

        # subtle random colour shift
        shift = np.random.randint(-40, 40, roi.shape, dtype=np.int16)
        roi = np.clip(roi.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        image[y:y+size, x:x+size] = roi
        return image


class BlurAlteration(Alteration):
    """
    Applies Gaussian blur to the region.
    """

    def apply(self, image, region):
        x, y, size = region.x, region.y, region.size
        roi = image[y:y+size, x:x+size]

        blurred = cv2.GaussianBlur(roi, (11, 11), 0)
        image[y:y+size, x:x+size] = blurred
        return image


class BrightnessAlteration(Alteration):
    """
    Adjusts brightness slightly.
    """

    def apply(self, image, region):
        x, y, size = region.x, region.y, region.size
        roi = image[y:y+size, x:x+size]

        value = random.randint(-50, 50)
        roi = np.clip(roi + value, 0, 255).astype(np.uint8)

        image[y:y+size, x:x+size] = roi
        return image