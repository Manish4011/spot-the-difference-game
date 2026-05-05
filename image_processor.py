import cv2
import random
from difference_region import DifferenceRegion
from alteration import (
    ColorShiftAlteration,
    BlurAlteration,
    BrightnessAlteration
)


class ImageProcessor:
    """
    Main engine that:
    1. Loads an image
    2. Creates a clone
    3. Generates 5 NON-OVERLAPPING regions
    4. Applies random alterations
    """

    def __init__(self, num_differences: int = 5, region_size: int = 50):
        self.num_differences = num_differences
        self.region_size = region_size

        # Polymorphism: storing different alteration objects together
        self.alterations = [
            ColorShiftAlteration(region_size),
            BlurAlteration(region_size),
            BrightnessAlteration(region_size)
        ]

    # ---------- IMAGE LOADING ----------
    def load_image(self, path: str):
        image = cv2.imread(path)

        if image is None:
            raise ValueError("Could not load image. Check file path.")

        return image

    # ---------- GENERATE NON-OVERLAPPING REGIONS ----------
    def generate_regions(self, width: int, height: int):
        regions = []

        while len(regions) < self.num_differences:
            x = random.randint(0, width - self.region_size)
            y = random.randint(0, height - self.region_size)

            new_region = DifferenceRegion(x, y, self.region_size)

            # Check overlap with existing regions
            overlap = False
            for region in regions:
                if new_region.overlaps(region):
                    overlap = True
                    break

            if not overlap:
                regions.append(new_region)

        return regions

    # ---------- APPLY RANDOM ALTERATIONS ----------
    def apply_differences(self, image, regions):
        modified = image.copy()

        for region in regions:
            alteration = random.choice(self.alterations)  # polymorphism
            modified = alteration.apply(modified, region)

        return modified

    # ---------- MAIN PROCESS ----------
    def process_image(self, path: str):
        """
        Returns:
            original image,
            modified image,
            list of DifferenceRegion objects
        """
        original = self.load_image(path)
        h, w = original.shape[:2]

        regions = self.generate_regions(w, h)
        modified = self.apply_differences(original, regions)

        return original, modified, regions