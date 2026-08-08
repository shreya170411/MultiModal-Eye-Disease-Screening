import os
from collections import defaultdict

# ==========================================
# DATASET PATH
# ==========================================
dataset_path = r"E:\Datasets\AMD_binary"
splits = ["train", "val", "test"]

expected_counts = {
    "train": {"0": 2300, "1": 2300},
    "val": {"0": 350, "1": 350},
    "test": {"0": 350, "1": 350}
}

print("=" * 70)
print("VERIFYING AMD_BINARY DATASET")
print("=" * 70)

for split in splits:
    split_path = os.path.join(dataset_path, split)
    print(f"\nChecking Split: {split.upper()}")
    
    if not os.path.exists(split_path):
        print(f"❌ Missing split folder: {split}")
        continue
    
    class_folders = [
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ]
    
    print(f"Found label folders: {class_folders}")
    
    total_images = 0
    
    for label in sorted(class_folders):
        label_path = os.path.join(split_path, label)
        
        images = [
            f for f in os.listdir(label_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        count = len(images)
        total_images += count
        
        expected = expected_counts[split].get(label, "N/A")
        status = "✅ MATCH" if count == expected else "⚠ MISMATCH"
        
        meaning = "NORMAL" if label == "0" else "AMD" if label == "1" else "UNKNOWN"
        
        print(f"Label {label} ({meaning}) → {count} images (Expected: {expected}) {status}")
    
    print(f"Total images in {split}: {total_images}")

print("\n" + "=" * 70)
print("AMD_BINARY VERIFICATION COMPLETE")
print("=" * 70)
