import os
import random
import shutil

random.seed(42)

# source folders
fundus_src = r"E:\Datasets\fundus_unified\images"
oct_src = r"E:\Datasets\oct_unified\images"
slit_src = r"E:\Datasets\ded_unified\images"

# output
root = r"E:\Major_Eye\modality_dataset"

classes = {
    "fundus": fundus_src,
    "oct": oct_src,
    "slitlamp": slit_src
}

train_count = 300
val_count = 60


def copy_subset(src, dst, n):

    imgs = [f for f in os.listdir(src) if f.lower().endswith(("jpg","png","jpeg"))]

    sample = random.sample(imgs, n)

    os.makedirs(dst, exist_ok=True)

    for f in sample:
        shutil.copy(
            os.path.join(src,f),
            os.path.join(dst,f)
        )


for cls,src in classes.items():

    train_dst = os.path.join(root,"train",cls)
    val_dst = os.path.join(root,"val",cls)

    copy_subset(src,train_dst,train_count)
    copy_subset(src,val_dst,val_count)


print("Dataset created at:",root)