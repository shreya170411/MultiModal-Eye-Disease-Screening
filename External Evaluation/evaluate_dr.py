import os
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score

# =========================
# DEVICE & SETTINGS
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Reduce batch size for CPU
BATCH_SIZE = 8 if DEVICE == "cpu" else 16

# =========================
# PATHS
# =========================
BASE = "E:/Major_Eye"
EYE_IMG = "E:/Datasets/EyePACS/resized_train_cropped/resized_train_cropped"
EYE_CSV = "E:/Datasets/EyePACS/trainLabels_cropped.csv"
MESSIDOR_IMG = "E:/Datasets/Messidor2/messidor-2/messidor-2/preprocess"
MESSIDOR_CSV = "E:/Datasets/Messidor2/messidor_data.csv"
MODEL_PATH = BASE + "/Results/dr_results/final_dr_model_best.pth"

# =========================
# DATASET CLASS
# =========================
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
            img = Image.new("RGB", (300, 300))
        img = self.transform(img)
        label = torch.tensor(row.label).float()
        return img, label

# =========================
# TRANSFORMS
# =========================
tfms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# =========================
# LOAD MODEL
# =========================
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=1)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()
print("Model loaded.")

# =========================
# LOAD EYEPACS (INTERNAL TEST)
# =========================
print("Loading EyePACS data...")
eyepacs = pd.read_csv(EYE_CSV)
eyepacs["path"] = eyepacs["image"].apply(lambda x: f"{EYE_IMG}/{x}.jpeg")
eyepacs["label"] = (eyepacs["level"] > 0).astype(int)
eyepacs = eyepacs.sample(n=1000, random_state=42)
print(f"EyePACS sample size: {len(eyepacs)}")

# =========================
# LOAD MESSIDOR (EXTERNAL TEST)
# =========================
print("Loading Messidor data...")
messidor = pd.read_csv(MESSIDOR_CSV)
messidor = messidor[messidor["adjudicated_gradable"] == 1]
messidor["path"] = messidor["id_code"].apply(lambda x: f"{MESSIDOR_IMG}/{x}")
messidor["label"] = (messidor["diagnosis"] > 0).astype(int)
messidor["exists"] = messidor["path"].apply(os.path.exists)
messidor = messidor[messidor["exists"] == True]
print(f"Messidor size (gradable + existing): {len(messidor)}")

# =========================
# DATALOADERS
# =========================
internal_loader = DataLoader(DRDataset(eyepacs, tfms), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
external_loader = DataLoader(DRDataset(messidor, tfms), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =========================
# METRICS FUNCTION
# =========================
def evaluate(loader, threshold, desc=""):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for img, label in tqdm(loader, desc=desc, leave=False):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            out = torch.sigmoid(model(img).squeeze())
            preds = (out > threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(out.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, sen, spe, prec, f1, auc, cm

# =========================
# FIND BEST THRESHOLD ON INTERNAL SET
# =========================
print("\nOptimising threshold on internal EyePACS sample...")
thresholds = np.arange(0.3, 0.7, 0.02)
best_th = 0.5
best_acc = 0
for th in thresholds:
    acc, _, _, _, _, _, _ = evaluate(internal_loader, th, desc=f"Threshold {th:.2f}")
    if acc > best_acc:
        best_acc = acc
        best_th = th
print(f"\n🔥 BEST THRESHOLD: {best_th:.2f} (Accuracy: {best_acc:.4f})")

# =========================
# FINAL RESULTS
# =========================
def print_results(name, loader):
    print(f"\n===== {name} =====")
    for th in [0.5, best_th]:
        acc, sen, spe, prec, f1, auc, cm = evaluate(loader, th, desc=f"Threshold {th:.2f}")
        print(f"\nThreshold: {th:.2f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {acc:.4f}")
        print(f"Sensitivity: {sen:.4f}")
        print(f"Specificity: {spe:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

print_results("INTERNAL (EyePACS sample)", internal_loader)
print_results("EXTERNAL (Messidor)", external_loader)