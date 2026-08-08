# Multi‑Modal Eye Disease Screening System

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-ff4b4b)](https://streamlit.io/)

**Automated detection of Diabetic Retinopathy, Glaucoma, Age‑related Macular Degeneration, and Dry Eye Disease from multi‑modal eye images**

</div>

---

## 📌 Overview

This repository contains the complete codebase for a **multi‑modal, multi‑disease eye screening system** that detects four major eye conditions:

- **Diabetic Retinopathy (DR)** – from fundus images
- **Glaucoma** – from fundus images
- **Age‑related Macular Degeneration (AMD)** – from OCT images
- **Dry Eye Disease (DED)** – from slit‑lamp images

The system uses **four disease‑specific expert models** (EfficientNetV2‑S, ConvNeXt‑Tiny, DenseNet‑121, EfficientNet‑B0) and compares **multiple fusion architectures** (Late Fusion with Logistic Regression/MLP, Weighted, Attention, Residual Attention, and Ablation), with the best performance achieved by **Residual Attention Fusion**.

---

## 🧠 Key Contributions

- ✅ **Four disease‑specific models** trained on curated datasets
- ✅ **Multiple fusion architectures** (Late, Weighted, Attention, Residual Attention, Ablation)
- ✅ **Feature extraction pipeline** for fusion training
- ✅ **External validation** on independent datasets (Messidor‑2, REFUGE2, OCT2017, Roboflow DED)
- ✅ **Streamlit web application** with quality checks, TTA, and educational profiles
- ✅ **Comprehensive ablation studies** (modality flags, attention mechanisms)
- ✅ **Publication‑ready figures** (ROC curves, confusion matrices, bar charts, tables)

---

## 📊 Results Summary

### Disease‑Specific Expert Models (Internal Test)

| Disease | Modality | Architecture | AUC | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|---------|----------|--------------|-----|--------------|-----------------|-----------------|
| DR      | Fundus   | EfficientNetV2‑S | 0.9166 | 85.7 | 81.5 | 87.1 |
| Glaucoma| Fundus   | ConvNeXt‑Tiny   | 0.9618 | 90.3 | 86.3 | 92.9 |
| AMD     | OCT      | DenseNet‑121    | 1.0000 | 100.0 | 100.0 | 100.0 |
| DED     | Slit‑lamp| EfficientNet‑B0 | 0.9208 | 90.4 | 85.3 | 93.9 |

### External Validation

| Disease | External Dataset | Threshold | Accuracy (%) | AUC | Sensitivity (%) | Specificity (%) |
|---------|------------------|-----------|--------------|-----|-----------------|-----------------|
| DR      | Messidor‑2       | 0.45      | 78.7         | 0.8440 | 74.6 | 81.6 |
| Glaucoma| REFUGE2          | 0.98      | 89.0         | 0.8995 | 80.0 | 90.0 |
| AMD     | OCT2017          | 0.5       | 99.9         | 0.99998 | 99.8 | 100.0 |
| DED     | Roboflow DED     | 0.539     | 63.3         | 0.7126 | 56.2 | 76.0 |

### Fusion Architecture Comparison (Macro AUC)

| Architecture | Macro AUC | Notebook |
|--------------|-----------|----------|
| Logistic Regression | 0.9903 | `Late_Fusion_Retrained_final`  |
| 1‑Layer MLP | 0.9880 | `Late_Fusion_Retrained_final` |
| 2‑Layer MLP | 0.9896 | `Late_Fusion_Retrained_final` |
| Weighted Fusion | 0.9884 | `Weighted_Fusion{Fusion_part}` |
| Attention Fusion | 0.9902 | `Attention_based_fusion(baseline)` |
| **Residual Attention Fusion** | **0.9915** | `Residual_Attention_Fusion` |
| Ablation (No Modality) | 0.9906 | `Attention_with_no_modality` |

---

## 📁 Repository Structure
```
MultiModal-Eye-Disease-Screening/
│
├── dataset_preparation/ # Dataset cleaning & preparation scripts
│ ├── FUNDUS/
│ │ ├── prepare_fundus_dataset.py
│ │ └── verify_fundus_dataset.py
│ ├── OCT/
│ │ ├── prepare_oct_dataset.py
│ │ └── verify_oct_dataset.py
│ ├── SLITLAMP/
│ │ ├── prepare_ded_dataset.py
│ │ └── verify_ded_unified_dataset.py
| ├── AMD/
│ │ ├── extract_amd_binary.py
│ │ ├── clean_amd_binary.py
│ │ ├── verify_amd_binary.py
│ │ ├── check_amd_dataset.py
│ │ ├── verify_amd_combined.py
│ │ ├── verify_oct2017.py
│ │ └── create_amd_combined.py
| ├── DED/
│ │ ├── analyze_ded_annotations.py
│ │ ├── verify_ded_dataset.py
│ │ ├── post_verify_ded.py
│ │ ├── clean_DED_small_ext.py
│ │ └── convert_ded_corrected.py
│ ├── GLAUCOMA/
│ │ ├── clean_glaucoma_dataset.py
│ │ └── check_glaucoma_dataset.py
│ ├── MODALITY_CLASSIFIER/
│ │ ├── create_modality_dataset.py
| | ├── check_modality_dataset.py
│ │ ├── train_modality_model.py
│ │ └── test_modality_model.py
│ └── MULTIMODAL_DATASET/
│ ├── create_multimodal_master_csv.py
│ └── split_multimodal_dataset.py
│
├── disease_models/ # Individual disease models
│ ├── DR/
│ │ └──  DR_training.py # DR training (EfficientNetV2‑S)
│ │  
│ ├── GlAUCOMA/
│ │ └──  GLAUCOMA_training.ipynb # Glaucoma training (ConvNeXt‑Tiny)
│ │  
│ ├── GLAUCOMA/
│ │ └──  AMD_training.ipynb # AMD training (DenseNet‑121)
│ │ 
│ └── DED/
│ | └──  DED_training.ipynb # DED training (EfficientNet‑B0)
│ │
├── feature_extraction/ # Feature extraction for fusion
│ ├── Train_feature_extraction.ipynb
│ ├── Val_feature_extraction.ipynb
│ └── Test_feature_extraction.ipynb
│
├── fusion_models/ # Fusion model training
│ ├── Late_Fusion_Retrained_final.ipynb # Late fusion (MLP)
│ ├── Weighted_Fusion{Fusion_part}.ipynb # Weighted fusion
│ ├── Attention_based_fusion(baseline).ipynb # Attention fusion
│ ├── Residual_Attention_Fusion.ipynb # ⭐ Best – residual attention 
│ └── Attention_with_no_modality.ipynb # Ablation (no modality flags)
│
├── external_evaluation/ # All external evaluation scripts
│ ├── evaluate_dr.py # Messidor‑2 evaluation and internal evaluation
│ ├── evaluate_glau_ext.py # REFUGE2 evaluation
│ ├── evaluate_AMD_ext.py # OCT2017 evaluation
│ └── evaluate_DED_ext.py # Roboflow DED evaluation
│
├── figures/ # Publication‑ready figures
│ ├── ROC_curves_all_diseases.png
│ ├── ablation_modality.png    # Comparison with modality and without modality (bar graph)
│ ├── ablation_per_disease.png   # Comparison of AUC with and without modality per disease (bar graph)
│ ├── ablation_table.png       # Comparison of AUC per disease with and without ablation
│ ├── confusion_matrices_optimal_all.png
│ ├── confusion_matrix_optimal_AMD.png
│ ├── confusion_matrix_optimal_DED.png
│ ├── confusion_matrix_optimal_DR.png
│ ├── confusion_matrix_optimal_Glaucoma.png
│ ├── confusion_matrices_standard_all.png
│ ├── confusion_matrix_standard_AMD.png
│ ├── confusion_matrix_standard_DED.png
│ ├── confusion_matrix_standard_DR.png
│ ├── confusion_matrix_standard_Glaucoma.png
│ ├── dataset_distribution.png
│ ├── dataset_distribution_fusion.png
│ ├── dataset_distribution_table.png
│ ├── external_validation_table.png
│ ├── fusion_comparison.png     # Late fusion (layer-wise comparison of MLP) - (bar graph)
│ ├── fusion_comparison_full.png   # Comparison of all fusion models (bar graph)
│ ├── fusion_comparison_table.png  # Comparison of all fusion models in table
│ ├── fusion_metrics_table.png     # Residual Fusion metrics (shows influence in each disease)
│ ├── individual_experts_comparison.png
│ ├── individual_experts_confusion_matrices.png
│ └── per_disease_AUC_bar.png
│
├── UI/ # Streamlit applications
│ ├── app_code.py # Main multi‑modal app
│ └──  disease_profiles.json
│
├── requirements.txt
├── .gitinore
└── README.md
```
---

## 🚀 Installation

### 1. Clone the repository
```
git clone https://github.com/shreya170411/MultiModal-Eye-Disease-Screening.git
cd MultiModal-Eye-Disease-Screening
```
### 2. Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```
### 3. Install dependencies
```
pip install -r requirements.txt
```
### 4. Download model weights
Download the trained .pth files from the links below and place them in the corresponding folders:

| Model | Link |
|-------|---------|
| **`DR (EfficientNetV2‑S)`** | [Download link](https://drive.google.com/file/d/11DkOkOaTcBNgMnbOTk4EZzP_oG5cZf79/view?usp=drive_link) |
| **`Glaucoma (ConvNeXt‑Tiny)`** | [Download link](https://drive.google.com/file/d/1LRfW5Llm2QHvvJURHtEJdssnpbJYaVmo/view?usp=drive_link) |
| **`AMD (DenseNet‑121)`** | [Download link](https://drive.google.com/file/d/17LGgvC9wJ8OzDlbnrZxn8TkO1w8PDXMI/view?usp=drive_link) |
| **`DED (EfficientNet‑B0)`** | [Download link](https://drive.google.com/file/d/1jA-1KCOLdjwcX6GWPMGMKbbXLl-JiPCn/view?usp=drive_link) |
| **`Modality (MobileNetV3)`** | [Download link](https://drive.google.com/file/d/1OTJsGJLsJGkKfFop2eG_exmQBDDnAx2e/view?usp=drive_link) |
| **`Late Fusion`** |[Download link](https://drive.google.com/file/d/1KY6INY_E4gA9LBlAbPuHFrIuQyYlT7dD/view?usp=drive_link) |
| **`Weighted Fusion`** | [Download link](https://drive.google.com/file/d/1L59s21LBo1DaFd2c7QHRxiphqSY0BhDV/view?usp=drive_link)|
| **`Attention Fusion`** | [Download link](https://drive.google.com/file/d/1I4CMrn1P34NkPo39TXoWt0jUqwmZT47z/view?usp=drive_link) |
| **`Residual Fusion`** | [Download link](https://drive.google.com/file/d/18jiewXsLUcypzUmIY4jUMOX7BHZByrP8/view?usp=drive_link) |
| **`Ablation study`** | [Download link](https://drive.google.com/file/d/1Yr8mEpnu9yE9YJMMzb3ni6_ut7yg_Mvt/view?usp=drive_link) |

---
## 🧪 Reproducing Results
### 1. Train Disease‑Specific Models
Open and run the respective codes in disease_models/:

- DR/DR_training.py
- GLAUCOMA/GLAUCOMA_training.ipynb
- AMD/AMD_training.ipynb
- DED/DED_training.ipynb

### 2. Extract Features for Fusion
Run the notebooks in feature_extraction/ in order:

- Train_feature_extraction.ipynb
- Val_feature_extraction.ipynb
- Test_feature_extraction.ipynb

### 3. Train Fusion Models
Run the notebooks in fusion_models/:

- Late_Fusion_Retrained_final.ipynb
- Weighted_Fusion{Fusion_part}.ipynb
- Attention_based_fusion(baseline).ipynb
- Residual_Attention_Fusion.ipynb (best model)
- Attention_with_no_modality.ipynb (ablation)

### 4. Evaluate on External Datasets
Run the scripts in external_evaluation/:
```
python external_evaluation/evaluate_dr_external.py
python external_evaluation/evaluate_glaucoma_external.py
python external_evaluation/evaluate_amd_external.py
python external_evaluation/evaluate_ded_external.py
```

### 5. Running the Web Application
Multi‑Modal App:
```
streamlit run UI/appy_code.py
```

## 📊 Datasets
| Dataset | Modality | Purpose | Source |
|---------|----------|---------|--------|
APTOS 2019 + EyePACS (sampled) | Fundus |	DR training |	Kaggle |
Fundus Glaucoma Detection Data | Fundus |	Glaucoma training	| Kaggle |
Retinal OCT Image Classification - C8 (AMD + Normal) + OCT2017 (sampled) | OCT | AMD training |	Kaggle |
Dry Eye Prediction Computer Vision Model |	Slit‑lamp	| DED training | Roboflow |
Messidor‑2	| Fundus | DR external validation |	Kaggle |
REFUGE2 |	Fundus | Glaucoma external validation	| Kaggle |
OCT2017	| OCT |	AMD external validation |	Kaggle |
dry eye Computer Vision Dataset | DED |	DED external validation	| Roboflow |

## ⚠️ Disclaimer

This system is intended for educational and research purposes only. Despite demonstrating strong generalisation capability on multiple independent datasets, it is not a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified ophthalmologist or healthcare provider for any eye‑related concerns. The authors assume no responsibility for any clinical decisions made based on the outputs of this system.

## 🙏 Acknowledgements
- Kaggle for hosting the datasets
- Google Colab for free GPU access
- The open‑source community for PyTorch, timm, and Streamlit

<div align="center">-------------- For research and educational purposes only. -------------- </div> 
