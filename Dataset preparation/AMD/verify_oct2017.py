import os
from PIL import Image
import numpy as np
from collections import defaultdict

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATASET_PATH = r"E:\Datasets\OCT2017"
EXPECTED_CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
SPLITS = ["train", "val", "test"]

# --------------------------------------------------
# VERIFICATION
# --------------------------------------------------

print("="*70)
print("VERIFYING OCT2017 DATASET")
print("="*70)

total_images = 0
corrupted_images = 0
grayscale_images = 0
small_images = 0

widths = []
heights = []

for split in SPLITS:
    print(f"\nChecking Split: {split.upper()}")
    split_path = os.path.join(DATASET_PATH, split)

    if not os.path.exists(split_path):
        print(f"❌ Missing split folder: {split_path}")
        continue

    found_classes = os.listdir(split_path)
    print("Found label folders:", found_classes)

    for cls in EXPECTED_CLASSES:
        class_path = os.path.join(split_path, cls)

        if not os.path.exists(class_path):
            print(f"❌ Missing class folder: {cls}")
            continue

        images = os.listdir(class_path)
        image_count = len(images)
        total_images += image_count

        print(f"Label {cls} → {image_count} images")

        for img_name in images:
            img_path = os.path.join(class_path, img_name)

            try:
                with Image.open(img_path) as img:
                    img.verify()

                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size

                    widths.append(width)
                    heights.append(height)

                    if width < 100 or height < 100:
                        small_images += 1

                    if img.mode == "L":
                        grayscale_images += 1

            except Exception:
                corrupted_images += 1

print("\n" + "="*70)
print("DATASET HEALTH REPORT")
print("="*70)
print("Total Images Checked:", total_images)
print("Corrupted Images:", corrupted_images)
print("Very Small Images (<100px):", small_images)
print("Grayscale Images (after conversion check):", grayscale_images)

if widths and heights:
    print("\nResolution Statistics:")
    print("Min Width:", min(widths))
    print("Max Width:", max(widths))
    print("Min Height:", min(heights))
    print("Max Height:", max(heights))
    print("Average Width:", int(np.mean(widths)))
    print("Average Height:", int(np.mean(heights)))

print("\nIntegrity Check Complete.")
print("="*70)
