import torch
import timm
import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# ---------- CONFIG ----------
MODEL_PATH = r"E:\Major_Eye\Results\ded_results\best_ded_model.pth"
TEST_DIR = r"E:\Datasets\DED_small_external"   # contains 0/ and 1/
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# ---------- TRANSFORMS (same as training) ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------- LOAD MODEL ----------
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------- LOAD DATA ----------
dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- INFERENCE ----------
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images).view(-1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

# ---------- METRICS ----------
y_true = np.array(all_labels)
y_prob = np.array(all_probs)

# Use optimal threshold from your internal validation (0.539)
threshold = 0.539
y_pred = (y_prob > threshold).astype(int)

auc = roc_auc_score(y_true, y_prob)
acc = accuracy_score(y_true, y_pred)
sens = recall_score(y_true, y_pred)
spec = recall_score(y_true, y_pred, pos_label=0)
prec = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("=== DED External Test (DED_small_external) ===")
print(f"Test set size: {len(y_true)}")
print(f"Class distribution: Normal={sum(y_true==0)}, DED={sum(y_true==1)}")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity: {sens:.4f}")
print(f"Specificity: {spec:.4f}")
print(f"Precision: {prec:.4f}")
print(f"F1: {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")