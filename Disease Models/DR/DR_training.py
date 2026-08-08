import os
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score

# =======================
# PATHS
# =======================
APTOS_PATH = "E:/datasets/APTOS"
EYE_PATH = "E:/datasets/EyePACS"
MESSIDOR_PATH = "E:/datasets/Messidor2"

MODEL_PATH = "E:/Major_Eye/Results/dr_results/final_dr_model_best.pth"
SAVE_DIR = "E:/Major_Eye/Results/dr_results"

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =======================
# LOAD DATA (UNCHANGED)
# =======================
aptos = pd.read_csv(f"{APTOS_PATH}/train.csv")
aptos["path"] = aptos["id_code"].apply(lambda x: f"{APTOS_PATH}/train_images/{x}.png")
aptos["label"] = (aptos["diagnosis"] > 0).astype(int)

eyepacs = pd.read_csv(f"{EYE_PATH}/trainLabels_cropped.csv")
eyepacs["path"] = eyepacs["image"].apply(
    lambda x: f"{EYE_PATH}/resized_train_cropped/resized_train_cropped/{x}.jpeg"
)
eyepacs["label"] = (eyepacs["level"] > 0).astype(int)

# =======================
# SAMPLING (UNCHANGED)
# =======================
pos = eyepacs[eyepacs.label == 1]
neg = eyepacs[eyepacs.label == 0]

pos_sample = pos.sample(frac=0.4, random_state=42)
neg_sample = neg.sample(n=len(pos_sample)*2, random_state=42)

eyepacs_sampled = pd.concat([pos_sample, neg_sample])

df = pd.concat([
    aptos[["path","label"]],
    eyepacs_sampled[["path","label"]]
])

pos = df[df.label == 1]
neg = df[df.label == 0]

neg_sample = neg.sample(n=int(len(pos)*1.2), random_state=42)
df_balanced = pd.concat([pos, neg_sample]).reset_index(drop=True)

# =======================
# DATASET
# =======================
class DRDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            img = Image.open(row.path).convert("RGB")
        except:
            img = Image.new("RGB", (300,300))

        img = self.transform(img)
        label = torch.tensor(row.label).float()

        return img, label

# =======================
# TRANSFORMS
# =======================
val_tfms = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =======================
# LOAD MODEL
# =======================
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=1)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

# =======================
# MESSIDOR TEST
# =======================
messidor = pd.read_csv(f"{MESSIDOR_PATH}/messidor_data.csv")
messidor = messidor[messidor["adjudicated_gradable"] == 1]

messidor["path"] = messidor["id_code"].apply(
    lambda x: f"{MESSIDOR_PATH}/messidor-2/messidor-2/preprocess/{x}"
)
messidor["label"] = (messidor["diagnosis"] > 0).astype(int)

messidor = messidor[messidor["path"].apply(os.path.exists)]

test_loader = DataLoader(
    DRDataset(messidor, val_tfms),
    batch_size=16,
    shuffle=False
)

# =======================
# TTA FUNCTION (UNCHANGED)
# =======================
def tta_predict(img):
    out1 = torch.sigmoid(model(img)).view(-1)

    img_flip = torch.flip(img, dims=[3])
    out2 = torch.sigmoid(model(img_flip)).view(-1)

    img_bright = torch.clamp(img * 1.1, 0, 1)
    out3 = torch.sigmoid(model(img_bright)).view(-1)

    return (out1 + out2 + out3) / 3.0

# =======================
# FINAL EVALUATION
# =======================
THRESHOLD = 0.45

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for img, label in test_loader:
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        out = tta_predict(img)
        preds = (out > THRESHOLD).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_probs.extend(out.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

accuracy = (tn + tp)/(tn+fp+fn+tp)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
f1 = (2*tp)/(2*tp+fp+fn)
auc = roc_auc_score(all_labels, all_probs)

print("\n🔥 FINAL DR RESULTS")
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("F1:", f1)
print("AUC:", auc)
print("Confusion Matrix:\n", cm)

# =======================
# SAVE RESULTS
# =======================
pd.DataFrame({
    "accuracy":[accuracy],
    "sensitivity":[sensitivity],
    "specificity":[specificity],
    "f1":[f1],
    "auc":[auc]
}).to_csv(f"{SAVE_DIR}/dr_final_results.csv", index=False)

print("✅ Done")