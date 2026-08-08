import os
from collections import defaultdict

# ============================
# Dataset Path
# ============================
dataset_path = r"E:\Datasets\Glaucoma_pytorch"

splits = ["train", "val", "test"]

print("=" * 60)
print("GLAUCOMA DATASET STRUCTURE VERIFICATION")
print("=" * 60)

for split in splits:
    split_path = os.path.join(dataset_path, split)
    
    print(f"\nChecking split: {split.upper()}")
    
    if not os.path.exists(split_path):
        print(f"❌ {split} folder NOT found!")
        continue
    
    class_counts = defaultdict(int)
    total_images = 0
    
    # List class folders
    class_folders = [
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ]
    
    print(f"Found class folders: {class_folders}")
    
    for class_label in class_folders:
        class_path = os.path.join(split_path, class_label)
        
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        count = len(image_files)
        class_counts[class_label] = count
        total_images += count
    
    # Print class distribution
    for label, count in class_counts.items():
        if label == "0":
            meaning = "Glaucoma NOT Present"
        elif label == "1":
            meaning = "Glaucoma Present"
        else:
            meaning = "⚠ Unknown Label"
        
        print(f"Label {label} ({meaning}) → {count} images")
    
    print(f"Total images in {split}: {total_images}")

print("\n" + "=" * 60)
print("Verification Complete")
print("=" * 60)
