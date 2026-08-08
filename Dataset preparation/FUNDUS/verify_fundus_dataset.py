import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import hashlib

# ===========================
# PATHS
# ===========================

DATASET_ROOT = r"E:\Datasets\fundus_unified"
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
CSV_PATH = os.path.join(DATASET_ROOT, "fundus_labels.csv")

print("="*60)
print("FUNDUS DATASET INTEGRITY CHECK")
print("="*60)

# ===========================
# LOAD CSV
# ===========================

df = pd.read_csv(CSV_PATH)
print(f"\nCSV rows: {len(df)}")

# ===========================
# IMAGE FILE COUNT
# ===========================

all_images = [f for f in os.listdir(IMAGE_DIR)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Images in folder: {len(all_images)}")

# ===========================
# CHECK 1: CSV vs FOLDER COUNT
# ===========================

if len(df) == len(all_images):
    print("✅ CSV count matches image folder count.")
else:
    print("❌ Mismatch between CSV rows and image files!")

# ===========================
# CHECK 2: MISSING FILES
# ===========================

missing_files = []
for path in df["image_path"]:
    full_path = os.path.join(DATASET_ROOT, path)
    if not os.path.exists(full_path):
        missing_files.append(path)

print(f"\nMissing files from CSV: {len(missing_files)}")

# ===========================
# CHECK 3: EXTRA FILES NOT IN CSV
# ===========================

csv_filenames = set([os.path.basename(p) for p in df["image_path"]])
folder_filenames = set(all_images)

extra_files = folder_filenames - csv_filenames

print(f"Extra files in folder not in CSV: {len(extra_files)}")

# ===========================
# CHECK 4: DUPLICATE ENTRIES
# ===========================

duplicates = df[df.duplicated(subset=["image_path"], keep=False)]
print(f"Duplicate CSV entries: {len(duplicates)}")

# ===========================
# CHECK 5: MULTI-LABEL CONFLICT
# ===========================

df["label_sum"] = df[["DR","Glaucoma","AMD","DED"]].sum(axis=1)
multi_label_conflicts = df[df["label_sum"] > 1]

print(f"Multi-label conflicts (>1 positive): {len(multi_label_conflicts)}")

# ===========================
# CHECK 6: CORRUPTED IMAGES
# ===========================

corrupted = []

print("\nChecking for corrupted images...")

for img_name in tqdm(all_images):
    img_path = os.path.join(IMAGE_DIR, img_name)
    try:
        with Image.open(img_path) as img:
            img.verify()
    except:
        corrupted.append(img_name)

print(f"Corrupted images: {len(corrupted)}")

# ===========================
# CHECK 7: CLASS DISTRIBUTION
# ===========================

print("\nClass Distribution:")
print(df[["DR","Glaucoma","AMD","DED"]].sum())

print("\nLabel Value Counts:")
print("DR:")
print(df["DR"].value_counts())
print("\nGlaucoma:")
print(df["Glaucoma"].value_counts())

# ===========================
# FINAL SUMMARY
# ===========================

print("\n" + "="*60)
print("DATASET INTEGRITY SUMMARY")
print("="*60)

if (len(missing_files)==0 and
    len(extra_files)==0 and
    len(duplicates)==0 and
    len(multi_label_conflicts)==0 and
    len(corrupted)==0 and
    len(df)==len(all_images)):
    
    print("🎉 DATASET IS CLEAN AND READY FOR FUSION TRAINING.")
else:
    print("⚠ Issues detected. Review above outputs carefully.")