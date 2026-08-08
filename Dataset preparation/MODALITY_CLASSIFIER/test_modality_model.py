import os
import random
import torch
import timm
from PIL import Image
from torchvision import transforms

DEVICE = "cpu"

model_path = r"E:\Major_Eye\modality\modality_model.pth"
data_root = r"E:\Major_Eye\modality_dataset\val"

classes = ["fundus","oct","slitlamp"]

# ------------------------------------------------
# Transform
# ------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ------------------------------------------------
# Load model
# ------------------------------------------------

model = timm.create_model(
    "mobilenetv3_small_050",
    pretrained=False,
    num_classes=3
)

model.load_state_dict(torch.load(model_path,map_location=DEVICE))
model.eval()

print("\nModel loaded.\n")

# ------------------------------------------------
# Test random images
# ------------------------------------------------

correct = 0
total = 0

for cls in classes:

    folder = os.path.join(data_root,cls)

    images = os.listdir(folder)

    sample = random.sample(images,10)

    print("\n===== Testing:",cls,"=====\n")

    for img_name in sample:

        path = os.path.join(folder,img_name)

        img = Image.open(path).convert("RGB")

        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():

            out = model(tensor)

            probs = torch.softmax(out,dim=1)[0]

            pred_idx = torch.argmax(probs).item()

        pred_class = classes[pred_idx]

        confidence = probs[pred_idx].item()

        print(f"{img_name}")
        print("GT:",cls," | Pred:",pred_class," | Conf:",round(confidence,3))

        if pred_class == cls:
            correct += 1

        total += 1

        print()

# ------------------------------------------------
# Accuracy
# ------------------------------------------------

acc = correct/total*100

print("\n======================")
print("Accuracy:",round(acc,2),"%")
print("======================\n")