import os
from collections import defaultdict

# =====================================
# CONFIGURATION
# =====================================
dataset_path = r"E:\Datasets\AMD\RetinalOCT_Dataset\RetinalOCT_Dataset"
splits = ["train", "val", "test"]

target_classes = ["AMD", "NORMAL"]

# Expected counts (based on your info)
expected_counts = {
    "train": {"AMD": 2300, "NORMAL": 2300},
    "val": {"AMD": 350, "NORMAL": 350},
    "test": {"AMD": 350, "NORMAL": 350}
}

print("=" * 70)
print("AMD DATASET STRUCTURE VERIFICATION (Binary Extraction)")
print("=" * 70)

for split in splits:
    split_path = os.path.join(dataset_path, split)
    
    print(f"\nChecking Split: {split.upper()}")
    
    if not os.path.exists(split_path):
        print(f"❌ {split} folder NOT found!")
        continue
    
    class_folders = [
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ]
    
    print(f"All disease folders found: {class_folders}")
    
    total_images = 0
    
    for cls in target_classes:
        class_path = os.path.join(split_path, cls)
        
        if not os.path.exists(class_path):
            print(f"❌ Missing required folder: {cls}")
            continue
        
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        count = len(image_files)
        total_images += count
        
        expected = expected_counts[split][cls]
        
        status = "✅ MATCH" if count == expected else "⚠ MISMATCH"
        
        print(f"{cls} → {count} images (Expected: {expected}) {status}")
    
    print(f"Total (AMD + NORMAL) in {split}: {total_images}")

print("\n" + "=" * 70)
print("Verification Complete")
print("=" * 70)
