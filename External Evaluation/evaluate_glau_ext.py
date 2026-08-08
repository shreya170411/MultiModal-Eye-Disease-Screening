import torch
import timm
import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# ---------- CONFIG ----------
MODEL_PATH = r"E:\Major_Eye\Results\Glaucoma_results\best_model.pth"
LABEL_CSV_PATH = r"E:\Dataset2\REFUGE2\REFUGE2\test_labels.csv"
USE_FULL_PATH_FROM_CSV = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# ---------- TRANSFORMS ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------- CUSTOM DATASET ----------
class GlaucomaExternalDataset(Dataset):
    def __init__(self, csv_path, transform=None, use_full_path=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.use_full_path = use_full_path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path'] if self.use_full_path else row['image_path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row['glaucoma_label']
        return image, torch.tensor(label, dtype=torch.long)

# ---------- LOAD DATASET AND LOADER ----------
dataset = GlaucomaExternalDataset(LABEL_CSV_PATH, transform=transform, use_full_path=USE_FULL_PATH_FROM_CSV)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- LOAD MODEL ----------
model = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------- INFERENCE ----------
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

y_true = np.array(all_labels)
y_prob = np.array(all_probs)

# ---------- THRESHOLD EVALUATION ----------
thresholds = np.arange(0.50, 1.00, 0.01)
results = []
best_f1 = 0
best_f1_th = 0.5
best_youden = 0
best_youden_th = 0.5
best_equal_th = 0.5
best_equal_diff = 1.0

for th in thresholds:
    preds = (y_prob > th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn) if (tp+fn) > 0 else 0
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0
    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
    youden = sens + spec - 1
    
    results.append((th, sens, spec, prec, f1, youden))
    
    if f1 > best_f1:
        best_f1 = f1
        best_f1_th = th
    if youden > best_youden:
        best_youden = youden
        best_youden_th = th
    # find threshold where sens and spec are closest
    diff = abs(sens - spec)
    if diff < best_equal_diff:
        best_equal_diff = diff
        best_equal_th = th

# ---------- PRINT SUMMARY ----------
print("\n=== Threshold Optimisation Results ===")
print(f"Max F1-score: {best_f1:.4f} at threshold {best_f1_th:.2f}")
print(f"Max Youden index: {best_youden:.4f} at threshold {best_youden_th:.2f}")
print(f"Sens ≈ Spec (closest): threshold {best_equal_th:.2f} (diff={best_equal_diff:.4f})")

# ---------- EVALUATE EACH CANDIDATE ----------
def evaluate_at_threshold(th, name):
    preds = (y_prob > th).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, preds)
    sens = recall_score(y_true, preds)
    spec = recall_score(y_true, preds, pos_label=0)
    prec = precision_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    print(f"\n--- {name} (threshold = {th:.2f}) ---")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

print("\n=== REFUGE2 External Test – Multiple Operating Points ===")
evaluate_at_threshold(best_f1_th, "Max F1")
evaluate_at_threshold(best_youden_th, "Max Youden")
evaluate_at_threshold(best_equal_th, "Sens ≈ Spec")
# Also include the earlier specificity‑driven threshold (0.98) for comparison
evaluate_at_threshold(0.98, "Specificity ≥ 90% (threshold 0.98)")

# Optional: save results to CSV for further analysis
df_results = pd.DataFrame(results, columns=['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Youden'])
df_results.to_csv("threshold_analysis_refuge2.csv", index=False)
print("\nDetailed threshold analysis saved to 'threshold_analysis_refuge2.csv'")