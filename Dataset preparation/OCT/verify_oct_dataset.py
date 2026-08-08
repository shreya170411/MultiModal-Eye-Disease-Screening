import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATASET_ROOT = r"E:\Datasets\oct_unified"
IMAGE_DIR = os.path.join(DATASET_ROOT, "images")
CSV_PATH = os.path.join(DATASET_ROOT, "oct_labels.csv")

print("="*60)
print("OCT DATASET INTEGRITY CHECK")
print("="*60)

df = pd.read_csv(CSV_PATH)

print("\nCSV rows:", len(df))

all_images = [f for f in os.listdir(IMAGE_DIR)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

print("Images in folder:", len(all_images))

# Count check
if len(df) == len(all_images):
    print("✅ CSV count matches image folder.")
else:
    print("❌ Mismatch detected.")

# Missing files
missing = []
for path in df["image_path"]:
    full_path = os.path.join(DATASET_ROOT, path)
    if not os.path.exists(full_path):
        missing.append(path)

print("Missing files:", len(missing))

# Extra files
csv_names = set([os.path.basename(p) for p in df["image_path"]])
folder_names = set(all_images)

extra = folder_names - csv_names
print("Extra files:", len(extra))

# Duplicate entries
duplicates = df[df.duplicated(subset=["image_path"], keep=False)]
print("Duplicate CSV entries:", len(duplicates))

# Multi-label conflict
df["label_sum"] = df[["DR","Glaucoma","AMD","DED"]].sum(axis=1)
conflicts = df[df["label_sum"] > 1]
print("Multi-label conflicts:", len(conflicts))

# Corruption check
print("\nChecking corrupted images...")
corrupted = []
for img in tqdm(all_images):
    try:
        with Image.open(os.path.join(IMAGE_DIR, img)) as im:
            im.verify()
    except:
        corrupted.append(img)

print("Corrupted images:", len(corrupted))

# Distribution
print("\nClass distribution:")
print(df[["DR","Glaucoma","AMD","DED"]].sum())

print("\nAMD value counts:")
print(df["AMD"].value_counts())

print("\nIntegrity Summary:")
if (len(missing)==0 and
    len(extra)==0 and
    len(duplicates)==0 and
    len(conflicts)==0 and
    len(corrupted)==0 and
    len(df)==len(all_images)):
    
    print("🎉 OCT dataset is CLEAN.")
else:
    print("⚠ Issues detected.")