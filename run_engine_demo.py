import os
import cv2
from image_processor import ImageProcessor


def ensure_output_folder():
    """Create outputs folder if it doesn't exist."""
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


def main():
    print("Starting Spot-The-Difference Engine Demo")

    ensure_output_folder()

    # change this to test different images
    image_path = "sample_images/test1.jpg"

    processor = ImageProcessor()

    # Run the engine
    original, modified, regions = processor.process_image(image_path)

    print("\nGenerated Difference Regions:")
    for i, r in enumerate(regions, start=1):
        print(f"{i}. {r}")

    # Save outputs (REQUIRED FOR ASSIGNMENT ZIP)
    cv2.imwrite("outputs/original.png", original)
    cv2.imwrite("outputs/modified.png", modified)

    print("\nImages saved to /outputs folder")

    # Optional preview window
    cv2.imshow("Original", original)
    cv2.imshow("Modified", modified)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()