import os
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import *

# ============================================
# CONFIG
# ============================================

DATASET_PATH = r"E:\Datasets\OCT2017\test"
MODEL_PATH = r"E:\Major_Eye\Results\amd_results\best_model_combined.pth"   # update if needed
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================
# TRANSFORMS (same as validation)
# ============================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ============================================
# CUSTOM DATASET (IMPORTANT)
# ============================================

class OCTBinaryDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        self.samples = []

        class_map = {
            "NORMAL": 0,
            "CNV": 1,
            "DRUSEN": 1
        }

        for cls in os.listdir(root):
            if cls not in class_map:
                continue  # skip DME

            cls_path = os.path.join(root, cls)

            for img in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img)
                self.samples.append((img_path, class_map[cls]))

        print("Total valid samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================================
# LOAD DATA
# ============================================

dataset = OCTBinaryDataset(DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================
# LOAD MODEL
# ============================================

model = timm.create_model("densenet121", pretrained=False)
in_features = model.classifier.in_features
model.classifier = torch.nn.Linear(in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ============================================
# INFERENCE
# ============================================

all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs).flatten()
all_labels = np.array(all_labels)

# ============================================
# METRICS FUNCTION
# ============================================

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return acc, prec, rec, specificity, f1, auc

# ============================================
# DEFAULT THRESHOLD
# ============================================

default_thresh = 0.5
default_preds = (all_probs > default_thresh).astype(int)

default_metrics = compute_metrics(all_labels, default_preds, all_probs)

# ============================================
# BEST THRESHOLD (ROC)
# ============================================

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]

best_preds = (all_probs > best_thresh).astype(int)

best_metrics = compute_metrics(all_labels, best_preds, all_probs)

# ============================================
# METRICS TABLE
# ============================================

columns = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]

df = pd.DataFrame([
    default_metrics,
    best_metrics
], index=["Threshold=0.5", f"Best Threshold={best_thresh:.4f}"], columns=columns)

print("\n===== OCT2017 EXTERNAL RESULTS =====")
print(df)

df.to_csv("oct2017_metrics.csv")

# ============================================
# CONFUSION MATRIX
# ============================================

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_cm(all_labels, default_preds, "OCT2017 (Threshold = 0.5)")
plot_cm(all_labels, best_preds, f"OCT2017 (Best Threshold = {best_thresh:.4f})")

# ============================================
# FINAL PRINT
# ============================================

print("\nBest Threshold:", round(best_thresh, 4))