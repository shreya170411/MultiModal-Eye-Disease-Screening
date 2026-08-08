import os
import shutil
import pandas as pd
from tqdm import tqdm

# ===============================
# PATHS
# ===============================

SOURCE_ROOT = r"E:\Datasets\DED_binary"
OUTPUT_ROOT = r"E:\Datasets\ded_unified"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_CSV = os.path.join(OUTPUT_ROOT, "ded_labels.csv")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

records = []

print("="*60)
print("BUILDING DED UNIFIED DATASET")
print("="*60)

# ==========================================================
# Use TRAIN + VALID only (Exclude TEST)
# ==========================================================

for split in ["train", "valid"]:
    split_path = os.path.join(SOURCE_ROOT, split)

    for class_label in ["0", "1"]:
        class_dir = os.path.join(split_path, class_label)

        if not os.path.exists(class_dir):
            continue

        for img_name in tqdm(os.listdir(class_dir), desc=f"{split} class {class_label}"):
            src_img = os.path.join(class_dir, img_name)

            new_name = f"DED_{split}_{class_label}_{img_name}"
            dst_img = os.path.join(OUTPUT_IMG_DIR, new_name)

            shutil.copy2(src_img, dst_img)

            ded_label = int(class_label)

            records.append([f"images/{new_name}", "slitlamp", 0, 0, 0, ded_label])

# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(records, columns=[
    "image_path",
    "modality",
    "DR",
    "Glaucoma",
    "AMD",
    "DED"
])

df.to_csv(OUTPUT_CSV, index=False)

print("\nDED dataset completed.")
print("Total images:", len(df))
print("CSV saved at:", OUTPUT_CSV)