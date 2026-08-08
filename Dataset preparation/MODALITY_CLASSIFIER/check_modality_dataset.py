import os
from PIL import Image
from collections import defaultdict

root = r"E:\Major_Eye\modality_dataset"

splits = ["train","val"]
classes = ["fundus","oct","slitlamp"]

print("\n===== DATASET STRUCTURE CHECK =====\n")

total_images = 0
corrupt = 0

resolution_stats = defaultdict(set)

for split in splits:

    print(f"\n--- {split.upper()} ---")

    for cls in classes:

        path = os.path.join(root,split,cls)

        if not os.path.exists(path):
            print(f"{cls}: ❌ folder missing")
            continue

        files = os.listdir(path)

        img_files = [
            f for f in files
            if f.lower().endswith(("jpg","jpeg","png"))
        ]

        print(f"{cls}: {len(img_files)} images")

        total_images += len(img_files)

        min_w,min_h = 99999,99999
        max_w,max_h = 0,0

        for f in img_files:

            fp = os.path.join(path,f)

            try:

                img = Image.open(fp)
                w,h = img.size

                resolution_stats[cls].add((w,h))

                min_w = min(min_w,w)
                min_h = min(min_h,h)

                max_w = max(max_w,w)
                max_h = max(max_h,h)

                if w < 128 or h < 128:
                    print("⚠ Very small image:",fp,w,h)

            except:
                print("❌ Corrupt image:",fp)
                corrupt += 1

        print(f"   Min resolution: {min_w}x{min_h}")
        print(f"   Max resolution: {max_w}x{max_h}")

print("\n===============================")
print("TOTAL IMAGES:",total_images)
print("CORRUPT IMAGES:",corrupt)
print("===============================\n")



print("===== RESOLUTION DISTRIBUTION =====\n")

for cls,res in resolution_stats.items():

    print(cls)

    for r in sorted(res)[:10]:
        print(" ",r)

    if len(res) > 10:
        print("  ...",len(res),"unique resolutions")

    print()