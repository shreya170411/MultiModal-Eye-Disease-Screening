import os
import shutil
import pandas as pd
from tqdm import tqdm

# ===============================
# PATHS
# ===============================

APTOS_PATH = r"E:\Datasets\aptos2019"
EYEPACS_PATH = r"E:\Datasets\EyePACS"
GLAUCOMA_PATH = r"E:\Datasets\Glaucoma_pytorch_backup"

OUTPUT_ROOT = r"E:\Major_Eye\fundus_unified"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_CSV = os.path.join(OUTPUT_ROOT, "fundus_labels.csv")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

records = []

# ==========================================================
# 1️⃣ APTOS 2019 (TRAIN ONLY)
# ==========================================================

print("Processing APTOS...")

aptos_csv = pd.read_csv(os.path.join(APTOS_PATH, "train.csv"))
aptos_img_dir = os.path.join(APTOS_PATH, "train_images")

for _, row in tqdm(aptos_csv.iterrows(), total=len(aptos_csv)):
    img_id = row["id_code"]
    label = row["diagnosis"]

    src_img = os.path.join(aptos_img_dir, img_id + ".png")
    new_name = f"DR_aptos_{img_id}.png"
    dst_img = os.path.join(OUTPUT_IMG_DIR, new_name)

    if not os.path.exists(src_img):
        continue

    shutil.copy2(src_img, dst_img)

    # Binary DR conversion
    dr_label = 1 if label > 0 else 0

    records.append([f"images/{new_name}", "fundus", dr_label, 0, 0, 0])

# ==========================================================
# 2️⃣ EYEPACS (resized_train_cropped ONLY)
# ==========================================================

print("Processing EyePACS...")

eyepacs_csv = pd.read_csv(os.path.join(EYEPACS_PATH, "trainLabels_cropped.csv"))
eyepacs_img_dir = os.path.join(EYEPACS_PATH, "resized_train_cropped")

for _, row in tqdm(eyepacs_csv.iterrows(), total=len(eyepacs_csv)):
    img_id = row["image"]
    label = row["level"]

    src_img = os.path.join(eyepacs_img_dir, img_id + ".jpeg")
    new_name = f"DR_eyepacs_{img_id}.jpeg"
    dst_img = os.path.join(OUTPUT_IMG_DIR, new_name)

    if not os.path.exists(src_img):
        continue

    shutil.copy2(src_img, dst_img)

    dr_label = 1 if label > 0 else 0

    records.append([f"images/{new_name}", "fundus", dr_label, 0, 0, 0])

# ==========================================================
# 3️⃣ GLAUCOMA (TRAIN + VAL ONLY)
# ==========================================================

print("Processing Glaucoma...")

for split in ["train", "val"]:
    split_path = os.path.join(GLAUCOMA_PATH, split)

    for class_label in ["0", "1"]:
        class_dir = os.path.join(split_path, class_label)

        if not os.path.exists(class_dir):
            continue

        for img_name in tqdm(os.listdir(class_dir)):
            src_img = os.path.join(class_dir, img_name)

            new_name = f"GLAU_{split}_{class_label}_{img_name}"
            dst_img = os.path.join(OUTPUT_IMG_DIR, new_name)

            shutil.copy2(src_img, dst_img)

            glaucoma_label = int(class_label)

            records.append([f"images/{new_name}", "fundus", 0, glaucoma_label, 0, 0])

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

print("=================================")
print("Fundus dataset preparation DONE.")
print("Total images:", len(df))
print("CSV saved at:", OUTPUT_CSV)
print("=================================")