import os

dataset_path = r"E:\Datasets\DED_binary"
splits = ["train", "valid", "test"]

print("=" * 70)
print("FINAL DED_BINARY VERIFICATION")
print("=" * 70)

total = 0

for split in splits:
    print(f"\nChecking Split: {split.upper()}")
    
    split_path = os.path.join(dataset_path, split)
    
    for label in ["0", "1"]:
        label_path = os.path.join(split_path, label)
        
        images = [
            f for f in os.listdir(label_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        count = len(images)
        total += count
        
        meaning = "NORMAL" if label == "0" else "DED"
        print(f"Label {label} ({meaning}) → {count}")
    
    print("-" * 40)

print("\nTotal images in DED_binary:", total)
print("=" * 70)
