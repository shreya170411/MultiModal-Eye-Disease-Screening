# ==========================================================
# DED_SMALL -> EXTERNAL BINARY DATASET CREATOR
# Use FULL dataset for External Testing Only
#
# Source:
# E:\Datasets\DED_small
#
# Output:
# E:\Datasets\DED_small_external
#
# Labels:
# 0 = NORMAL
# 1 = DED   (right / left)
# ==========================================================

import os
import shutil
from collections import defaultdict

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
SOURCE_PATH = r"E:\Datasets\DED_small"
DEST_PATH   = r"E:\Datasets\DED_small_external"

splits = ["train", "test"]

# ----------------------------------------------------------
# START
# ----------------------------------------------------------
print("=" * 80)
print("CREATING DED_SMALL EXTERNAL BINARY DATASET")
print("=" * 80)

# Remove old folder if exists
if os.path.exists(DEST_PATH):
    shutil.rmtree(DEST_PATH)

# Create output folders
os.makedirs(os.path.join(DEST_PATH, "0"), exist_ok=True)
os.makedirs(os.path.join(DEST_PATH, "1"), exist_ok=True)

# ----------------------------------------------------------
# CLASS RULES
# YOLO Labels:
# 0 = right
# 1 = left
# 2 = normal
#
# Binary:
# right / left = DED (1)
# normal       = NORMAL (0)
# ----------------------------------------------------------

counts = defaultdict(int)
copied_names = set()
multi_object_images = 0
total_processed = 0

for split in splits:

    print(f"\nProcessing {split.upper()} ...")

    images_path = os.path.join(SOURCE_PATH, split, "images")
    labels_path = os.path.join(SOURCE_PATH, split, "labels")

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"Skipping {split} (folder missing)")
        continue

    for label_file in os.listdir(labels_path):

        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(labels_path, label_file)

        with open(label_path, "r") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]

        if len(lines) == 0:
            continue

        # detect multiple annotations
        if len(lines) > 1:
            multi_object_images += 1

        image_name = os.path.splitext(label_file)[0]

        # locate image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            temp = os.path.join(images_path, image_name + ext)
            if os.path.exists(temp):
                img_path = temp
                break

        if img_path is None:
            continue

        # --------------------------------------
        # IMAGE LEVEL LABEL
        # if any class 0 or 1 => DED
        # else NORMAL
        # --------------------------------------
        binary_label = 0

        for line in lines:
            cls = line.split()[0]

            if cls in ["0", "1"]:
                binary_label = 1
                break

        # Avoid duplicate names
        base_name = os.path.basename(img_path)

        if base_name in copied_names:
            new_name = split + "_" + base_name
        else:
            new_name = base_name

        copied_names.add(new_name)

        save_path = os.path.join(
            DEST_PATH,
            str(binary_label),
            new_name
        )

        shutil.copy2(img_path, save_path)

        counts[binary_label] += 1
        total_processed += 1

# ----------------------------------------------------------
# FINAL REPORT
# ----------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL EXTERNAL DATASET SUMMARY")
print("=" * 80)

print(f"0 (NORMAL) : {counts[0]}")
print(f"1 (DED)    : {counts[1]}")
print(f"TOTAL      : {total_processed}")
print(f"Multi-object images detected : {multi_object_images}")

print("\nSaved To:")
print(DEST_PATH)

print("\nDataset Structure:")
print("DED_small_external/")
print("   0  -> NORMAL")
print("   1  -> DED")

print("\nREADY FOR EXTERNAL TESTING")
print("=" * 80)