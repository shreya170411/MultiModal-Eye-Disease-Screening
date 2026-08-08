import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# PATHS
# ===============================

MASTER_CSV = r"E:\Datasets\multimodal_unified\multimodal_labels.csv"
OUTPUT_DIR = r"E:\Datasets\multimodal_unified"

print("="*60)
print("CREATING STRATIFIED 70/15/15 SPLIT")
print("="*60)

df = pd.read_csv(MASTER_CSV)

# ==========================================================
# CREATE STRATIFICATION LABEL
# ==========================================================

def get_strat_label(row):
    if row["DR"] == 1:
        return "DR"
    elif row["Glaucoma"] == 1:
        return "Glaucoma"
    elif row["AMD"] == 1:
        return "AMD"
    elif row["DED"] == 1:
        return "DED"
    else:
        return "Normal"

df["strat_label"] = df.apply(get_strat_label, axis=1)

# ==========================================================
# FIRST SPLIT: TRAIN (70%) vs TEMP (30%)
# ==========================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["strat_label"],
    random_state=42
)

# ==========================================================
# SECOND SPLIT: VAL (15%) vs TEST (15%)
# ==========================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["strat_label"],
    random_state=42
)

# ==========================================================
# SAVE SPLITS
# ==========================================================

train_df.drop(columns=["strat_label"]).to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"), index=False
)

val_df.drop(columns=["strat_label"]).to_csv(
    os.path.join(OUTPUT_DIR, "val.csv"), index=False
)

test_df.drop(columns=["strat_label"]).to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"), index=False
)

print("\nSplit Complete.")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

print("\nTrain Distribution:")
print(train_df["strat_label"].value_counts())

print("\nVal Distribution:")
print(val_df["strat_label"].value_counts())

print("\nTest Distribution:")
print(test_df["strat_label"].value_counts())