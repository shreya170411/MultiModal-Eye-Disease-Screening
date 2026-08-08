import os
from PIL import Image
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
dataset_path = r"E:\Datasets\AMD_binary"
splits = ["train", "val", "test"]

remove_corrupted = False  # ⚠ Keep False initially

print("=" * 70)
print("AMD_BINARY DATASET INTEGRITY CHECK")
print("=" * 70)

total_images = 0
corrupted_images = []
grayscale_images = []
small_images = []
resolution_stats = []

for split in splits:
    split_path = os.path.join(dataset_path, split)
    print(f"\nChecking Split: {split.upper()}")
    
    for label in os.listdir(split_path):
        label_path = os.path.join(split_path, label)
        
        if not os.path.isdir(label_path):
            continue
        
        for file in os.listdir(label_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            
            img_path = os.path.join(label_path, file)
            total_images += 1
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                # Reopen after verify
                with Image.open(img_path) as img:
                    width, height = img.size
                    resolution_stats.append((width, height))
                    
                    # Check grayscale
                    if img.mode != "RGB":
                        grayscale_images.append(img_path)
                    
                    # Check small images
                    if width < 100 or height < 100:
                        small_images.append(img_path)
            
            except Exception:
                corrupted_images.append(img_path)
                if remove_corrupted:
                    os.remove(img_path)

# ==========================================
# REPORT
# ==========================================
print("\n" + "=" * 70)
print("DATASET HEALTH REPORT")
print("=" * 70)

print(f"Total Images Checked: {total_images}")
print(f"Corrupted Images: {len(corrupted_images)}")
print(f"Grayscale Images: {len(grayscale_images)}")
print(f"Very Small Images (<100px): {len(small_images)}")

if resolution_stats:
    widths = [r[0] for r in resolution_stats]
    heights = [r[1] for r in resolution_stats]
    
    print("\nResolution Statistics:")
    print(f"Min Width: {min(widths)}")
    print(f"Max Width: {max(widths)}")
    print(f"Min Height: {min(heights)}")
    print(f"Max Height: {max(heights)}")
    print(f"Average Width: {int(np.mean(widths))}")
    print(f"Average Height: {int(np.mean(heights))}")

print("\n" + "=" * 70)

if corrupted_images:
    print("\nSample Corrupted Files:")
    for img in corrupted_images[:5]:
        print(img)

if grayscale_images:
    print("\nSample Grayscale Images:")
    for img in grayscale_images[:5]:
        print(img)

if small_images:
    print("\nSample Small Images:")
    for img in small_images[:5]:
        print(img)

print("\nIntegrity Check Complete.")
print("=" * 70)
