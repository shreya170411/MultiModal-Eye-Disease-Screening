import os
import shutil
import random
from collections import defaultdict

# ==========================================
# PATHS
# ==========================================
source_path = r"E:\Datasets\DED"
destination_path = r"E:\Datasets\DED_binary"

splits = ["train", "valid", "test"]  # note: folder name is 'valid'

print("=" * 70)
print("CONVERTING DED → BINARY (CORRECTED MAPPING, TRAIN BALANCED)")
print("=" * 70)

# Remove existing folder
if os.path.exists(destination_path):
    shutil.rmtree(destination_path)

os.makedirs(destination_path, exist_ok=True)

for split in splits:
    print(f"\nProcessing Split: {split.upper()}")
    
    images_path = os.path.join(source_path, split, "images")
    labels_path = os.path.join(source_path, split, "labels")
    
    class_images = defaultdict(list)
    
    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue
        
        label_path = os.path.join(labels_path, label_file)
        
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        image_name = os.path.splitext(label_file)[0]
        
        # Find matching image
        image_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(images_path, image_name + ext)
            if os.path.exists(candidate):
                image_file = candidate
                break
        
        if image_file is None:
            continue
        
        # ==========================================
        # CORRECT CLASS MAPPING
        # YOLO: class 0 = DED
        #       class 1 = NORMAL
        # ==========================================
        
        image_label = 0  # default NORMAL
        
        for line in lines:
            class_id = line.split()[0]
            if class_id == "0":   # DED detected
                image_label = 1   # set classification label as DED
                break
        
        class_images[image_label].append(image_file)
    
    num_normal = len(class_images[0])
    num_ded = len(class_images[1])
    
    print(f"Original → Normal: {num_normal}, DED: {num_ded}")
    
    # Create destination folders
    for label in ["0", "1"]:
        os.makedirs(os.path.join(destination_path, split, label), exist_ok=True)
    
    # Balance only TRAIN
    if split == "train":
        min_count = min(num_normal, num_ded)
        
        balanced_normal = random.sample(class_images[0], min_count)
        balanced_ded = random.sample(class_images[1], min_count)
        
        print(f"Balanced TRAIN → Each Class: {min_count}")
        
        for img_path in balanced_normal:
            shutil.copy2(img_path,
                         os.path.join(destination_path, split, "0",
                                      os.path.basename(img_path)))
        
        for img_path in balanced_ded:
            shutil.copy2(img_path,
                         os.path.join(destination_path, split, "1",
                                      os.path.basename(img_path)))
    
    else:
        print(f"Keeping {split.upper()} original distribution")
        
        for img_path in class_images[0]:
            shutil.copy2(img_path,
                         os.path.join(destination_path, split, "0",
                                      os.path.basename(img_path)))
        
        for img_path in class_images[1]:
            shutil.copy2(img_path,
                         os.path.join(destination_path, split, "1",
                                      os.path.basename(img_path)))

print("\n" + "=" * 70)
print("DED BINARY DATASET CREATED SUCCESSFULLY (CORRECTED)")
print("=" * 70)
