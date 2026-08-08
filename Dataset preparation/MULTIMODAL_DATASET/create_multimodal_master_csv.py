import os
import pandas as pd

# ===============================
# SOURCE CSV PATHS
# ===============================

FUNDUS_CSV = r"E:\Datasets\fundus_unified\fundus_labels.csv"
OCT_CSV    = r"E:\Datasets\oct_unified\oct_labels.csv"
DED_CSV    = r"E:\Datasets\ded_unified\ded_labels.csv"

# ===============================
# ROOT IMAGE FOLDERS
# ===============================

FUNDUS_ROOT = r"E:\Datasets\fundus_unified"
OCT_ROOT    = r"E:\Datasets\oct_unified"
DED_ROOT    = r"E:\Datasets\ded_unified"

# ===============================
# OUTPUT
# ===============================

OUTPUT_DIR = r"E:\Datasets\multimodal_unified"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "multimodal_labels.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("CREATING MULTIMODAL MASTER CSV")
print("="*60)

# ==========================================================
# LOAD CSV FILES
# ==========================================================

fundus_df = pd.read_csv(FUNDUS_CSV)
oct_df    = pd.read_csv(OCT_CSV)
ded_df    = pd.read_csv(DED_CSV)

# ==========================================================
# CONVERT RELATIVE PATHS TO ABSOLUTE PATHS
# ==========================================================

fundus_df["image_path"] = fundus_df["image_path"].apply(
    lambda x: os.path.join(FUNDUS_ROOT, x)
)

oct_df["image_path"] = oct_df["image_path"].apply(
    lambda x: os.path.join(OCT_ROOT, x)
)

ded_df["image_path"] = ded_df["image_path"].apply(
    lambda x: os.path.join(DED_ROOT, x)
)

# ==========================================================
# CONCATENATE ALL
# ==========================================================

master_df = pd.concat([fundus_df, oct_df, ded_df], ignore_index=True)

# ==========================================================
# SAVE MASTER CSV
# ==========================================================

master_df.to_csv(OUTPUT_CSV, index=False)

print("\nMultimodal dataset created successfully.")
print("Total images:", len(master_df))
print("Saved at:", OUTPUT_CSV)

print("\nClass Distribution Summary:")
print(master_df[["DR","Glaucoma","AMD","DED"]].sum())

print("\nModality Distribution:")
print(master_df["modality"].value_counts())