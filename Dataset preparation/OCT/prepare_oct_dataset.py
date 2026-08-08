import os
import shutil
import pandas as pd
from tqdm import tqdm

# ===============================
# PATHS
# ===============================

SOURCE_ROOT = r"E:\Datasets\AMD_combined"
OUTPUT_ROOT = r"E:\Datasets\oct_unified"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_CSV = os.path.join(OUTPUT_ROOT, "oct_labels.csv")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

records = []

print("="*60)
print("BUILDING OCT UNIFIED DATASET")
print("="*60)

# ==========================================================
# Use TRAIN + VAL only (Exclude TEST)
# ==========================================================

for split in ["train", "val"]:
    split_path = os.path.join(SOURCE_ROOT, split)

    for class_label in ["0", "1"]:
        class_dir = os.path.join(split_path, class_label)

        if not os.path.exists(class_dir):
            continue

        for img_name in tqdm(os.listdir(class_dir), desc=f"{split} class {class_label}"):
            src_img = os.path.join(class_dir, img_name)

            new_name = f"OCT_{split}_{class_label}_{img_name}"
            dst_img = os.path.join(OUTPUT_IMG_DIR, new_name)

            shutil.copy2(src_img, dst_img)

            amd_label = int(class_label)

            # Multi-label vector: [DR, Glaucoma, AMD, DED]
            records.append([f"images/{new_name}", "oct", 0, 0, amd_label, 0])

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

print("\nOCT dataset completed.")
print("Total images:", len(df))
print("CSV saved at:", OUTPUT_CSV)