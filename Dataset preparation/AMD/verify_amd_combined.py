import os
from PIL import Image
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATASET_PATH = r"E:\Datasets\AMD_combined"
EXPECTED_SPLITS = ["train", "val", "test"]
EXPECTED_CLASSES = ["0", "1"]

# --------------------------------------------------
# VERIFICATION START
# --------------------------------------------------

print("="*75)
print("VERIFYING AMD_COMBINED DATASET")
print("="*75)

total_images = 0
corrupted_images = 0
small_images = 0

widths = []
heights = []

for split in EXPECTED_SPLITS:

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
        count = len(images)
        total_images += count

        label_name = "NORMAL" if cls == "0" else "AMD"

        print(f"Label {cls} ({label_name}) → {count} images")

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

            except Exception:
                corrupted_images += 1

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "="*75)
print("DATASET HEALTH REPORT")
print("="*75)

print("Total Images Checked:", total_images)
print("Corrupted Images:", corrupted_images)
print("Very Small Images (<100px):", small_images)

if widths and heights:
    print("\nResolution Statistics:")
    print("Min Width:", min(widths))
    print("Max Width:", max(widths))
    print("Min Height:", min(heights))
    print("Max Height:", max(heights))
    print("Average Width:", int(np.mean(widths)))
    print("Average Height:", int(np.mean(heights)))

# --------------------------------------------------
# CLASS DISTRIBUTION SUMMARY
# --------------------------------------------------

print("\n" + "="*75)
print("FINAL SPLIT DISTRIBUTION")
print("="*75)

for split in EXPECTED_SPLITS:
    split_path = os.path.join(DATASET_PATH, split)
    if not os.path.exists(split_path):
        continue

    print(f"\n{split.upper()}:")

    for cls in EXPECTED_CLASSES:
        class_path = os.path.join(split_path, cls)
        if os.path.exists(class_path):
            print(f"  Class {cls} → {len(os.listdir(class_path))}")

print("\nIntegrity Check Complete.")
print("="*75)
