class DifferenceRegion:
    """
    Represents a square region where a difference exists.
    This will be used later for click detection in the GUI.
    """

    def __init__(self, x: int, y: int, size: int):
        self.x = x
        self.y = y
        self.size = size
        self.found = False  # used later by GUI

    def contains_point(self, px: int, py: int) -> bool:
        """
        Check if a click point is inside this region.
        (GUI teammate will use this later)
        """
        return (
            self.x <= px <= self.x + self.size and
            self.y <= py <= self.y + self.size
        )

    def overlaps(self, other_region) -> bool:
        """
        Prevent overlapping differences.
        Required by assignment.
        """
        return not (
            self.x + self.size < other_region.x or
            other_region.x + other_region.size < self.x or
            self.y + self.size < other_region.y or
            other_region.y + other_region.size < self.y
        )

    def __repr__(self):
        return f"DifferenceRegion(x={self.x}, y={self.y}, size={self.size})"