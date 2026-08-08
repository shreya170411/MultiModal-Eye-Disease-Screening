import os
import shutil
import random

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

AMD_BINARY_PATH = r"E:/Datasets/AMD_binary"
OCT_PATH = r"E:/Datasets/OCT2017/train"
OUTPUT_PATH = r"E:/Datasets/AMD_combined"

SAMPLE_SIZE = 2000  # per class from OCT

random.seed(42)

# --------------------------------------------------
# CREATE FOLDER STRUCTURE
# --------------------------------------------------

for split in ["train", "val", "test"]:
    for cls in ["0", "1"]:
        os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

# --------------------------------------------------
# COPY AMD_BINARY FULLY
# --------------------------------------------------

print("Copying AMD_binary dataset...")

for split in ["train", "val", "test"]:
    for cls in ["0", "1"]:
        src = os.path.join(AMD_BINARY_PATH, split, cls)
        dst = os.path.join(OUTPUT_PATH, split, cls)

        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file), dst)

print("AMD_binary copied.")

# --------------------------------------------------
# SAMPLE OCT2017 TRAIN
# --------------------------------------------------

print("Sampling OCT2017...")

oct_classes = {
    "NORMAL": "0",
    "CNV": "1",
    "DRUSEN": "1"
}

for oct_class, target_class in oct_classes.items():

    class_path = os.path.join(OCT_PATH, oct_class)
    images = os.listdir(class_path)

    sampled_images = random.sample(images, SAMPLE_SIZE)

    for img in sampled_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(OUTPUT_PATH, "train", target_class)
        shutil.copy(src, dst)

    print(f"Sampled {SAMPLE_SIZE} images from {oct_class}")

print("OCT sampling complete.")

# --------------------------------------------------
# FINAL COUNT CHECK
# --------------------------------------------------

print("\nFinal TRAIN distribution:")

for cls in ["0", "1"]:
    path = os.path.join(OUTPUT_PATH, "train", cls)
    print(f"Class {cls}:", len(os.listdir(path)))

print("\nAMD_combined dataset created successfully.")
