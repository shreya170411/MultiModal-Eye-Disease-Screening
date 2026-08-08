import os
import shutil

# ==========================================
# SOURCE AND DESTINATION PATHS
# ==========================================
source_path = r"E:\Datasets\AMD\RetinalOCT_Dataset\RetinalOCT_Dataset"
destination_path = r"E:\Datasets\AMD_binary"

splits = ["train", "val", "test"]

# Class mapping
class_mapping = {
    "NORMAL": "0",
    "AMD": "1"
}

print("=" * 70)
print("EXTRACTING AMD BINARY DATASET")
print("=" * 70)

# Create destination root folder
os.makedirs(destination_path, exist_ok=True)

for split in splits:
    print(f"\nProcessing Split: {split.upper()}")
    
    for class_name, new_label in class_mapping.items():
        
        src_folder = os.path.join(source_path, split, class_name)
        dest_folder = os.path.join(destination_path, split, new_label)
        
        if not os.path.exists(src_folder):
            print(f"❌ Missing folder: {src_folder}")
            continue
        
        os.makedirs(dest_folder, exist_ok=True)
        
        images = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        
        print(f"{class_name} → Copying {len(images)} images to label {new_label}")
        
        for img in images:
            src_file = os.path.join(src_folder, img)
            dest_file = os.path.join(dest_folder, img)
            shutil.copy2(src_file, dest_file)

print("\n" + "=" * 70)
print("AMD BINARY DATASET EXTRACTION COMPLETE")
print("=" * 70)
