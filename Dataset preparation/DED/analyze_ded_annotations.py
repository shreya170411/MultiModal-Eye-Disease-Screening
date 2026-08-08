import os
from collections import defaultdict

# ==========================================
# DATASET PATH
# ==========================================
dataset_path = r"E:\Datasets\DED"
splits = ["train", "valid", "test"]

print("=" * 70)
print("DED ANNOTATION CONTENT ANALYSIS (YOLO FORMAT)")
print("=" * 70)

class_counts = defaultdict(int)
empty_label_files = []
multi_object_images = []
max_objects_in_image = 0
total_objects = 0
total_label_files = 0

for split in splits:
    print(f"\nChecking Split: {split.upper()}")
    
    labels_path = os.path.join(dataset_path, split, "labels")
    
    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue
        
        total_label_files += 1
        label_path = os.path.join(labels_path, label_file)
        
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) == 0:
            empty_label_files.append(label_file)
            continue
        
        num_objects = len(lines)
        total_objects += num_objects
        
        if num_objects > 1:
            multi_object_images.append(label_file)
        
        if num_objects > max_objects_in_image:
            max_objects_in_image = num_objects
        
        for line in lines:
            parts = line.split()
            class_id = parts[0]
            class_counts[class_id] += 1

# ==========================================
# REPORT
# ==========================================
print("\n" + "=" * 70)
print("ANNOTATION SUMMARY")
print("=" * 70)

print(f"Total Label Files: {total_label_files}")
print(f"Total Objects Annotated: {total_objects}")
print(f"Unique Class IDs Found: {list(class_counts.keys())}")

print("\nObject Count Per Class ID:")
for class_id, count in class_counts.items():
    print(f"Class {class_id} → {count} objects")

print(f"\nEmpty Label Files: {len(empty_label_files)}")
print(f"Images with Multiple Objects: {len(multi_object_images)}")
print(f"Max Objects in a Single Image: {max_objects_in_image}")

if empty_label_files:
    print("\nSample Empty Label Files:")
    for name in empty_label_files[:5]:
        print(name)

if multi_object_images:
    print("\nSample Multi-Object Images:")
    for name in multi_object_images[:5]:
        print(name)

print("\nAnalysis Complete.")
print("=" * 70)
