import os

# ==========================================
# DATASET PATH
# ==========================================
dataset_path = r"E:\Datasets\DED"
splits = ["train", "valid", "test"]

print("=" * 70)
print("DED DATASET STRUCTURE VERIFICATION (YOLO FORMAT)")
print("=" * 70)

total_images_all = 0
total_labels_all = 0

for split in splits:
    print(f"\nChecking Split: {split.upper()}")
    
    split_path = os.path.join(dataset_path, split)
    images_path = os.path.join(split_path, "images")
    labels_path = os.path.join(split_path, "labels")
    
    # Check folder existence
    if not os.path.exists(images_path):
        print(f"❌ Missing images folder in {split}")
        continue
    if not os.path.exists(labels_path):
        print(f"❌ Missing labels folder in {split}")
        continue
    
    # Get image files
    image_files = [
        f for f in os.listdir(images_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    
    # Get label files
    label_files = [
        f for f in os.listdir(labels_path)
        if f.lower().endswith('.txt')
    ]
    
    num_images = len(image_files)
    num_labels = len(label_files)
    
    total_images_all += num_images
    total_labels_all += num_labels
    
    print(f"Images: {num_images}")
    print(f"Labels: {num_labels}")
    
    # Check matching filenames
    image_names = set(os.path.splitext(f)[0] for f in image_files)
    label_names = set(os.path.splitext(f)[0] for f in label_files)
    
    missing_labels = image_names - label_names
    missing_images = label_names - image_names
    
    print(f"Missing Labels: {len(missing_labels)}")
    print(f"Missing Images: {len(missing_images)}")
    
    if missing_labels:
        print("Sample Missing Labels:")
        for name in list(missing_labels)[:5]:
            print(name)
    
    if missing_images:
        print("Sample Missing Images:")
        for name in list(missing_images)[:5]:
            print(name)

print("\n" + "=" * 70)
print("OVERALL DATASET SUMMARY")
print("=" * 70)

print(f"Total Images: {total_images_all}")
print(f"Total Labels: {total_labels_all}")

if total_images_all == 1127:
    print("✅ Total image count matches expected 1127")
else:
    print("⚠ Total image count does NOT match expected 1127")

print("\nVerification Complete.")
print("=" * 70)
