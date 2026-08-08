import streamlit as st
import torch
import timm
import json
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

DEVICE = "cpu"

# ---------------- PATHS ----------------
DR_MODEL = r"E:\Major_Eye\Results\dr_results\final_dr_model_best.pth"
GL_MODEL = r"E:\Major_Eye\Results\Glaucoma_results\best_model.pth"
AMD_MODEL = r"E:\Major_Eye\Results\amd_results\best_model_combined.pth"
DED_MODEL = r"E:\Major_Eye\Results\ded_results\best_ded_model.pth"
FUSION_MODEL = r"E:\Major_Eye\Results\residual_fusion_results_final\residual_fusion_model.pth"
MODALITY_MODEL = r"E:\Major_Eye\modality\modality_model.pth"
PROFILE_PATH = r"E:\Major_Eye\disease_profiles.json"

# ---------------- HELPER: FIX MOJIBAKE ----------------
def fix_encoding(text):
    if not isinstance(text, str):
        return text
    replacements = {
        'â€”': '—',
        'â€œ': '"',
        'â€': '"',
        'â€™': "'",
        'â€¦': '…',
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text

# ---------------- LOAD PROFILE WITH CLEANING ----------------
with open(PROFILE_PATH, 'r', encoding='utf-8') as f:
    raw_profiles = json.load(f)

disease_profiles = {}
for key, profile in raw_profiles.items():
    cleaned = {}
    for k, v in profile.items():
        if isinstance(v, str):
            cleaned[k] = fix_encoding(v)
        elif isinstance(v, list):
            cleaned[k] = [fix_encoding(item) if isinstance(item, str) else item for item in v]
        else:
            cleaned[k] = v
    disease_profiles[key] = cleaned

# ---------------- TRANSFORMS ----------------
fundus_transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

generic_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- TTA ----------------
tta_transforms = [
    lambda x: x,
    lambda x: transforms.functional.hflip(x),
    lambda x: transforms.functional.adjust_brightness(x,1.2),
    lambda x: transforms.functional.adjust_contrast(x,1.2),
]

# ---------------- FUSION MODEL ----------------
import torch.nn as nn

class ResidualAttentionFusion(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        attn = self.attention(x)
        weighted = x * (1 + 0.8 * attn)
        out = self.classifier(weighted)
        return out

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    dr = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=1)
    ckpt = torch.load(DR_MODEL, map_location=DEVICE)
    dr.load_state_dict(ckpt["model"])
    dr.eval()

    gl = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
    gl.load_state_dict(torch.load(GL_MODEL, map_location=DEVICE))
    gl.eval()

    amd = timm.create_model("densenet121", pretrained=False, num_classes=1)
    amd.load_state_dict(torch.load(AMD_MODEL, map_location=DEVICE))
    amd.eval()

    ded = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    ded.load_state_dict(torch.load(DED_MODEL, map_location=DEVICE))
    ded.eval()

    fusion = ResidualAttentionFusion()
    fusion.load_state_dict(torch.load(FUSION_MODEL, map_location=DEVICE))
    fusion.eval()

    modality = timm.create_model("mobilenetv3_small_050", pretrained=False, num_classes=3)
    modality.load_state_dict(torch.load(MODALITY_MODEL, map_location=DEVICE))
    modality.eval()

    return dr, gl, amd, ded, fusion, modality

dr_model, gl_model, amd_model, ded_model, fusion_model, modality_model = load_models()

# ---------------- QUALITY CHECK ----------------
def quality_check(img, modality=None):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    brightness = gray.mean()

    if modality == "fundus":
        if brightness < 20:
            return False, "Image too dark (fundus)"
        return True, "OK"

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 40:
        return False, "Image too blurry"
    if brightness < 25:
        return False, "Image too dark"
    return True, "OK"

# ---------------- MODALITY ----------------
def detect_modality(img):
    t = generic_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = torch.softmax(modality_model(t), dim=1)
    probs = out.numpy()[0]
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    top1 = np.max(probs)
    idx = np.argmax(probs)
    labels = ["fundus", "oct", "slitlamp"]
    if top1 < 0.80 or entropy > 0.8:
        return "invalid", top1
    return labels[idx], top1

def is_valid_medical_image(img):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.var() < 5:
        return False
    return True

def eye_domain_check(img, modality):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if modality == "fundus":
        return gray.var() > 10
    elif modality == "oct":
        return arr.std() < 100
    elif modality == "slitlamp":
        return arr.std() > 8
    return True

def is_oct_like(img):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if gray.var() < 3:
        return False
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    horizontal_strength = np.mean(np.abs(sobely))
    vertical_strength = np.mean(np.abs(sobelx))
    if horizontal_strength < vertical_strength * 0.7:
        return False
    return True

# ---------------- PREDICTIONS ----------------
def dr_predict(img):
    preds = []
    for aug in tta_transforms:
        aug_img = aug(img)
        t = fundus_transform(aug_img).unsqueeze(0)
        with torch.no_grad():
            out = dr_model(t)
            prob = torch.sigmoid(out).view(-1).item()
        preds.append(prob)
    return np.mean(preds)

def tta_predict(model, img, transform, mode="binary"):
    preds = []
    for aug in tta_transforms:
        aug_img = aug(img)
        t = transform(aug_img).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
            if mode == "binary2":
                out = torch.softmax(out, dim=1)
                val = out[:,1].item()
            else:
                val = torch.sigmoid(out).item()
        preds.append(val)
    return np.mean(preds)

def run_expert(img, modality):
    dr = gl = amd = ded = 0
    if modality == "fundus":
        dr = dr_predict(img)
        gl = tta_predict(gl_model, img, generic_transform, "binary2")
        dr = float(np.clip(dr, 0, 1))
        gl = float(np.clip(gl, 0, 1))
    elif modality == "oct":
        amd = tta_predict(amd_model, img, generic_transform)
    elif modality == "slitlamp":
        ded = tta_predict(ded_model, img, generic_transform)
    return dr, gl, amd, ded

def run_fusion(dr, gl, amd, ded, modality):
    flags = {"fundus":[1,0,0], "oct":[0,1,0], "slitlamp":[0,0,1]}
    f = flags[modality]
    x = torch.tensor([[dr, gl, amd, ded, f[0], f[1], f[2]]]).float()
    with torch.no_grad():
        out = torch.sigmoid(fusion_model(x)).numpy()[0]
    return out

# ==================== CORRECTED INTERPRETATION FUNCTIONS (from appy6.py) ====================
def interpret_fundus(dr_prob, gl_prob):
    dr_prob = float(np.clip(dr_prob, 0, 1))
    gl_prob = float(np.clip(gl_prob, 0, 1))
    
    # Disease detection logic – unchanged
    if dr_prob > 0.60 and dr_prob > gl_prob + 0.10:
        return "DR", dr_prob
    if gl_prob > 0.55 and gl_prob > dr_prob + 0.08:
        return "Glaucoma", gl_prob
    if dr_prob > 0.75 and gl_prob > 0.75 and abs(dr_prob - gl_prob) < 0.10:
        return "Uncertain (review manually)", max(dr_prob, gl_prob)
    
    # Normal case – FIXED: confidence = 1 - max disease probability
    max_disease = max(dr_prob, gl_prob)
    return "Fundus_Normal", 1 - max_disease

def interpret(probs, modality):
    dr, gl, amd, ded = probs
    if modality == "oct":
        # OCT logic – already correct
        if amd < 0.45:
            return "OCT_Normal", 1 - amd
        if amd > 0.65:
            return "AMD", amd
        return "OCT_Normal", 1 - amd
    if modality == "slitlamp":
        # Slitlamp logic – unchanged
        if ded > 0.85:
            return "DED", ded
        elif ded > 0.55:
            return "Possible DED", ded
        else:
            # Normal case – FIXED: confidence = 1 - ded
            return "Normal", 1 - ded
    return "Unknown", 0.0
# ============================================================================================

# ---------------- UI (UNCHANGED – YOUR APPROVED DESIGN) ----------------
st.set_page_config(
    page_title="Multi-Modal Eye Health Screening System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS (tight spacing + disclaimer style) ----------------
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.block-container{
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.main-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}

.sub-title {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 0.5rem;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    margin-top: 0;
}

.info-card {
    background: #f0f9ff;
    border-left: 6px solid #0284c7;
    padding: 16px;
    border-radius: 12px;
    margin: 1rem 0;
}

.result-good {
    background: #ecfdf5;
    border-left: 6px solid #10b981;
    padding: 14px;
    border-radius: 12px;
}

.result-warn {
    background: #fff7ed;
    border-left: 6px solid #f59e0b;
    padding: 14px;
    border-radius: 12px;
}

.result-bad {
    background: #fef2f2;
    border-left: 6px solid #ef4444;
    padding: 14px;
    border-radius: 12px;
}

.small-note {
    font-size: 0.88rem;
    color: #64748b;
}

.profile-section {
    margin-top: 1rem;
    padding-top: 0.5rem;
    border-top: 1px solid #e5e7eb;
}

/* ----- TIGHTEN SPACING INSIDE EXPANDER ----- */
.streamlit-expanderContent {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}

.streamlit-expanderContent li,
.stMarkdown li,
ul li,
ol li {
    margin-bottom: 2px !important;
    margin-top: 0px !important;
    line-height: 1.25 !important;
}

.streamlit-expanderContent ul,
.streamlit-expanderContent ol,
.stMarkdown ul,
.stMarkdown ol {
    margin-top: 0px !important;
    margin-bottom: 6px !important;
    padding-left: 20px !important;
}

.streamlit-expanderContent p,
.stMarkdown p {
    margin-bottom: 6px !important;
    margin-top: 0px !important;
    line-height: 1.3 !important;
}

.streamlit-expanderContent h1,
.streamlit-expanderContent h2,
.streamlit-expanderContent h3,
.streamlit-expanderContent h4,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4 {
    margin-bottom: 6px !important;
    margin-top: 8px !important;
}

.streamlit-expanderContent .stCaption {
    margin-top: 6px !important;
    margin-bottom: 0px !important;
}

/* ----- DISCLAIMER STYLING (larger font, colored background) ----- */
.disclaimer-text {
    font-size: 0.85rem;
    color: #6c757d;
    background-color: #f8f9fa;
    padding: 8px 12px;
    border-radius: 8px;
    margin-top: 12px;
    margin-bottom: 0;
    border-left: 3px solid #adb5bd;
    font-style: normal;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Multi‑Modal Eye Health Screening System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated analysis of Fundus, OCT, and Slit Lamp images for early detection of common eye diseases.</div>',
    unsafe_allow_html=True
)

# ---------------- BRIEF PROJECT INFO CARD ----------------
st.markdown("""
<div class="info-card">
    <b>📋 About this tool:</b><br>
    This system analyzes medical eye images (Fundus, OCT, or Slit Lamp) to screen for four common conditions:
    <b>Diabetic Retinopathy (DR)</b>, <b>Glaucoma</b>, <b>Age-related Macular Degeneration (AMD)</b>, and <b>Dry Eye Disease (DED)</b>.<br><br>
    Each image is first classified by modality, then routed to a dedicated expert model. A fusion network combines predictions for final interpretation.<br>
    <i>For screening support only – not a clinical diagnosis.</i>
</div>
""", unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Upload Eye Images")
st.markdown('<div class="small-note">Accepted formats: JPG, PNG, JPEG</div>', unsafe_allow_html=True)

files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

predict_btn = st.button("🔍 Analyze Images", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- COMPACT PROFILE WITH EXPANDER & STYLED DISCLAIMER (NO TRUNCATION) ----------------
def display_compact_profile(profile, is_disease=True):
    """Display short description + expandable details with full text (no truncation)."""
    if not profile:
        return
    
    st.markdown(f"**{profile.get('name', '')}**")
    st.info(profile.get("short_description", ""))
    
    with st.expander("📋 More details"):
        symptoms = profile.get("common_symptoms", [])
        if symptoms:
            st.markdown("**Common symptoms:**")
            for s in symptoms[:4]:
                st.markdown(f"• {s}")
        
        if is_disease:
            what = profile.get("what_is_it", "")
            why = profile.get("why_it_occurs", "")
            if what or why:
                st.markdown(f"**What & why:** {what} {why}")
            
            lifestyle = profile.get("lifestyle_considerations", [])
            early = profile.get("early_detection_importance", "")
            if lifestyle or early:
                st.markdown("**What you can do:**")
                if early:
                    st.markdown(f"• {early}")  # FULL TEXT
                for item in lifestyle[:2]:
                    st.markdown(f"• {item}")
            
            when = profile.get("when_to_seek_clinician", "")
            if when:
                st.markdown(f"**When to see a doctor:** {when}")
        else:
            risk = profile.get("risk_factors", [])
            if risk:
                st.markdown("**Risk factors to monitor:**")
                for r in risk[:2]:
                    st.markdown(f"• {r}")
            
            lifestyle = profile.get("lifestyle_considerations", [])
            early = profile.get("early_detection_importance", "")
            if lifestyle or early:
                st.markdown("**What you can do:**")
                if early:
                    st.markdown(f"• {early}")  # FULL TEXT
                for item in lifestyle[:2]:
                    st.markdown(f"• {item}")
            
            when = profile.get("when_to_seek_clinician", "")
            if when:
                st.markdown(f"**When to see a doctor:** {when}")
        
        disclaimer_text = profile.get("disclaimer", "")
        if disclaimer_text:
            st.markdown(f'<div class="disclaimer-text">⚠️ {disclaimer_text}</div>', unsafe_allow_html=True)

# ==========================================================
# BACKEND LOGIC (unchanged except for corrected interpret functions)
# ==========================================================

if predict_btn:

    if not files:
        st.warning("Please upload at least one image.")
        st.stop()

    for file in files:

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        try:
            img = Image.open(file).convert("RGB")
        except:
            st.error("Unable to read uploaded file.")
            st.markdown('</div>', unsafe_allow_html=True)
            continue

        left, right = st.columns([1.05, 1])

        with left:
            st.image(img, caption=file.name, use_container_width=True)

        with right:

            with st.spinner("Running analysis..."):

                # ---- Validation (unchanged) ----
                if not is_valid_medical_image(img):
                    st.markdown('<div class="result-bad"><b>Invalid Image</b><br>Not a valid medical eye image.</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                modality, mod_conf = detect_modality(img)

                if modality == "invalid":
                    st.markdown(f'<div class="result-bad"><b>Rejected</b><br>Low confidence modality detection ({mod_conf:.2f})</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                if modality == "oct":
                    if not is_oct_like(img):
                        st.markdown('<div class="result-bad"><b>Rejected</b><br>Image does not appear to be valid OCT.</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        continue

                if not eye_domain_check(img, modality):
                    st.markdown('<div class="result-bad"><b>Rejected</b><br>Unexpected image characteristics.</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                ok, msg = quality_check(img, modality)
                if not ok:
                    st.markdown(f'<div class="result-bad"><b>Rejected</b><br>{msg}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                st.markdown(f'<div class="result-good"><b>Detected Modality:</b> {modality.upper()} ({mod_conf*100:.1f}% confidence)</div>', unsafe_allow_html=True)

                # ---- Prediction ----
                dr, gl, amd, ded = run_expert(img, modality)

                if modality == "fundus":
                    disease, conf = interpret_fundus(dr, gl)
                    probs = [dr, gl, 0, 0]
                else:
                    fusion_probs = run_fusion(dr, gl, amd, ded, modality)
                    if max(fusion_probs) < 0.5:
                        probs = [dr, gl, amd, ded]
                        disease, conf = interpret(probs, modality)
                    else:
                        probs = fusion_probs
                        disease, conf = interpret(probs, modality)

                # ---- Result box ----
                if "Normal" in disease:
                    box = "result-good"
                elif "Possible" in disease or "Uncertain" in disease:
                    box = "result-warn"
                else:
                    box = "result-bad"

                st.markdown(f'<div class="{box}"><b>Screening Result:</b> {disease}<br>Confidence Score: {conf*100:.2f}%</div>', unsafe_allow_html=True)

                # ---- Compact Disease Profile with Expander ----
                profile_key = disease
                if disease == "Normal" and modality == "slitlamp":
                    profile_key = "Slitlamp_Normal"
                elif disease == "Normal" and modality == "fundus":
                    profile_key = "Fundus_Normal"
                elif disease == "Normal" and modality == "oct":
                    profile_key = "OCT_Normal"

                profile = disease_profiles.get(profile_key)

                if profile:
                    st.markdown('<div class="profile-section">', unsafe_allow_html=True)
                    is_disease_flag = not ("Normal" in disease or "Normal" in profile_key)
                    display_compact_profile(profile, is_disease=is_disease_flag)
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br>", unsafe_allow_html=True)
st.caption(
    "This platform is intended for screening support and educational demonstration only. "
    "Clinical diagnosis should be made by qualified ophthalmologists."
)