"""
HealthBandhu: An Explainable Hybrid AI Framework for Multiclass Disease Diagnosis,
Emergency Risk Assessment, and Clinical Decision Support
Developer: Joy Ghatak, B.Tech CSE, IEM newtown, UEM Kolkata
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import datetime
import io

# ── gdown: try import, install hint if missing ─────────────────
try:
    import gdown
    _HAS_GDOWN = True
except ModuleNotFoundError:
    _HAS_GDOWN = False

# ── PDF helper: try fpdf2 first, fall back to built-in writer ──
try:
    from fpdf import FPDF as _FPDF
    _USE_FPDF = True
except ModuleNotFoundError:
    _USE_FPDF = False


# ─────────────────────────────────────────────
# Google Drive Model Downloader
# ─────────────────────────────────────────────
def download_models_if_missing():
    """
    Download large model files from Google Drive on first run.
    Only downloads files that are not already present locally.
    Works on Streamlit Cloud, Colab, and local machines.
    """

    MODEL_FILES = {
        "models/healthbandhu_et.pkl":      "1goeklDxs7BwjA3ojBo8eGx-KiiFmSEl_",
        "models/healthbandhu_dnn.keras":   "1mPD4Ob1yG9ieZEnRXuON7LjqtvMLxkkS",
    }

    os.makedirs("models", exist_ok=True)

    if not _HAS_GDOWN:
        st.error(
            "⚠️ Package `gdown` is not installed. "
            "Add `gdown>=4.7.0` to your `requirements.txt` and redeploy."
        )
        st.stop()

    all_present = all(os.path.exists(p) for p in MODEL_FILES)
    if all_present:
        return

    st.info("⏳ First-time setup: Downloading AI models from Google Drive. This takes 1–2 minutes only once.")

    for local_path, file_id in MODEL_FILES.items():
        if os.path.exists(local_path):
            continue

        if "YOUR_" in file_id:
            st.error(
                f"❌ Google Drive File ID not set for `{local_path}`. "
                "Open `app.py`, find `MODEL_FILES`, and paste your real File ID."
            )
            st.stop()

        with st.spinner(f"📥 Downloading `{local_path}` from Google Drive..."):
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, local_path, quiet=False, fuzzy=True)

                if os.path.exists(local_path):
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    st.success(f"✅ Downloaded `{local_path}` ({size_mb:.1f} MB)")
                else:
                    st.error(
                        f"❌ Download failed for `{local_path}`. "
                        "Check that your Google Drive link is set to 'Anyone with the link'."
                    )
                    st.stop()

            except Exception as e:
                st.error(f"❌ Error downloading `{local_path}`: {e}")
                st.stop()

    st.success("✅ All models ready! Loading HealthBandhu...")
    st.rerun()


# Run downloader before anything else
download_models_if_missing()


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HealthBandhu – AI Disease Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS – Clean Hospital / Medical Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --blue-deep:   #0A2540;
    --blue-mid:    #1A6FA8;
    --teal:        #0D9488;
    --teal-light:  #CCFBF1;
    --white:       #FFFFFF;
    --gray-50:     #F8FAFC;
    --gray-100:    #F1F5F9;
    --gray-200:    #E2E8F0;
    --gray-600:    #475569;
    --gray-800:    #1E293B;
    --red-600:     #DC2626;
    --red-50:      #FEF2F2;
    --amber-500:   #F59E0B;
    --amber-50:    #FFFBEB;
    --green-600:   #16A34A;
    --green-50:    #F0FDF4;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--gray-800);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--blue-deep) 0%, #0D3B6B 100%);
    border-right: 1px solid #0F3460;
}
section[data-testid="stSidebar"] * { color: #E0EAF5 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

.sidebar-brand {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
    text-align: center;
}
.sidebar-brand h2 {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 0 0 4px 0;
    color: #FFFFFF !important;
}
.sidebar-brand p {
    font-size: 0.72rem;
    color: #94C9F0 !important;
    margin: 0;
    line-height: 1.4;
}

.sidebar-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
}
.sidebar-card h4 {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #5BB8F5 !important;
    margin: 0 0 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.10);
    padding-bottom: 6px;
}
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.78rem;
}
.sidebar-stat:last-child { border-bottom: none; }
.sidebar-stat .val {
    font-weight: 600;
    color: #5BB8F5 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
}

/* ── Top Header Bar ── */
.top-header {
    background: linear-gradient(135deg, var(--blue-deep) 0%, var(--blue-mid) 100%);
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 24px rgba(10,37,64,0.18);
}
.top-header-icon { font-size: 2.8rem; }
.top-header-text h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 4px 0;
    letter-spacing: -0.4px;
}
.top-header-text p {
    font-size: 0.82rem;
    color: #94C9F0;
    margin: 0;
}
.top-header-badge {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
}
.badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.70rem;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 0.5px;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--gray-200);
}
.section-header span.icon { font-size: 1.3rem; }
.section-header h2 {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--blue-deep);
    margin: 0;
    letter-spacing: -0.2px;
}
.section-header .tag {
    margin-left: auto;
    background: var(--teal-light);
    color: var(--teal);
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
}

/* ── Cards ── */
.metric-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
.metric-card .label {
    font-size: 0.70rem;
    font-weight: 600;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--blue-mid);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.68rem;
    color: var(--gray-600);
    margin-top: 4px;
}

/* ── Result Cards ── */
.result-main {
    background: linear-gradient(135deg, #EFF8FF 0%, #E0F2FE 100%);
    border: 1.5px solid #BAE6FD;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 16px;
}
.result-main .disease-name {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--blue-deep);
    margin-bottom: 6px;
    letter-spacing: -0.3px;
}
.result-main .confidence-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 10px;
}
.conf-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.conf-very-high  { background: #DCFCE7; color: #15803D; }
.conf-high       { background: #DBEAFE; color: #1D4ED8; }
.conf-moderate   { background: #FEF9C3; color: #854D0E; }
.conf-low        { background: #FFE4E6; color: #9F1239; }

/* ── Emergency Banners ── */
.emergency-critical {
    background: var(--red-50);
    border: 2px solid #FCA5A5;
    border-left: 6px solid var(--red-600);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.emergency-high {
    background: var(--amber-50);
    border: 2px solid #FCD34D;
    border-left: 6px solid var(--amber-500);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.emergency-normal {
    background: var(--green-50);
    border: 2px solid #86EFAC;
    border-left: 6px solid var(--green-600);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.emergency-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.emergency-body { font-size: 0.83rem; line-height: 1.6; }

/* ── Clinical Box ── */
.clinical-box {
    background: var(--gray-50);
    border: 1px solid var(--gray-200);
    border-radius: 12px;
    padding: 22px 26px;
    font-size: 0.84rem;
    line-height: 1.8;
    color: var(--gray-800);
}
.clinical-box h4 {
    font-size: 0.70rem;
    font-weight: 700;
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--gray-200);
    padding-bottom: 6px;
}

/* ── Architecture Box ── */
.arch-box {
    background: var(--blue-deep);
    border-radius: 14px;
    padding: 28px 32px;
    color: #E0EAF5;
}
.arch-box h4 {
    font-size: 0.70rem;
    font-weight: 700;
    color: #5BB8F5;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
}
.arch-step {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.arch-step:last-child { border-bottom: none; }
.arch-num {
    width: 30px; height: 30px;
    background: var(--blue-mid);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; color: #fff;
    flex-shrink: 0;
}
.arch-label { font-size: 0.88rem; font-weight: 500; }
.arch-desc  { font-size: 0.72rem; color: #94C9F0; margin-top: 2px; }
.arch-arrow { color: #5BB8F5; font-size: 1.1rem; margin-left: auto; }

/* ── Info box ── */
.info-box {
    background: #EFF8FF;
    border: 1px solid #BAE6FD;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.80rem;
    color: #1D4ED8;
    margin: 10px 0;
    line-height: 1.6;
}

/* ── Divider ── */
.hb-divider {
    border: none;
    border-top: 1px solid var(--gray-200);
    margin: 28px 0;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: var(--teal) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #0B7A6A !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--blue-mid) 0%, #1560A0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.3px !important;
    width: 100%;
    box-shadow: 0 2px 10px rgba(26,111,168,0.30) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 18px rgba(26,111,168,0.40) !important;
    transform: translateY(-1px) !important;
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background: var(--teal) !important;
    border-radius: 6px !important;
}

/* ── Dataframe ── */
.dataframe thead tr th {
    background: var(--blue-deep) !important;
    color: white !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Constants & Emergency Config
# ─────────────────────────────────────────────
EMERGENCY_WEIGHTS = {
    "shortness of breath":  5,
    "sharp chest pain":     5,
    "irregular heartbeat":  5,
    "fainting":             5,
    "seizures":             5,
    "breathing fast":       4,
    "chest tightness":      4,
    "throat swelling":      4,
}

MODEL_METRICS = {
    "Top-1 Accuracy":  85.70,
    "Top-3 Accuracy":  95.87,
    "Top-5 Accuracy":  97.92,
    "Top-10 Accuracy": 99.35,
}

CONFIDENCE_THRESHOLDS = {
    "Very High": 80,
    "High":      60,
    "Moderate":  40,
}


# ─────────────────────────────────────────────
# File Loading
# ─────────────────────────────────────────────
def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    errors = []
    artifacts = {}

    def _resolve(fname):
        for candidate in [f"models/{fname}", fname, f"/content/{fname}"]:
            if os.path.exists(candidate):
                return candidate
        return f"models/{fname}"

    files = {
        "symptoms":      _resolve("healthbandhu_symptoms.pkl"),
        "diseases":      _resolve("healthbandhu_diseases.pkl"),
        "label_encoder": _resolve("healthbandhu_label_encoder.pkl"),
        "metrics":       _resolve("healthbandhu_metrics.pkl"),
        "et_model":      _resolve("healthbandhu_et.pkl"),
    }

    for key, fname in files.items():
        if os.path.exists(fname):
            try:
                artifacts[key] = _load_pkl(fname)
            except Exception as e:
                errors.append(f"Could not load **{fname}**: {e}")
                artifacts[key] = None
        else:
            errors.append(f"File not found: **{fname}**")
            artifacts[key] = None

    dnn_path = _resolve("healthbandhu_dnn.keras")
    if os.path.exists(dnn_path):
        try:
            import tensorflow as tf
            artifacts["dnn_model"] = tf.keras.models.load_model(dnn_path)
        except Exception as e:
            errors.append(f"DNN model load failed (using Extra Trees fallback): {e}")
            artifacts["dnn_model"] = None
    else:
        errors.append(f"File not found: **{dnn_path}** (using Extra Trees fallback)")
        artifacts["dnn_model"] = None

    return artifacts, errors


# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
def build_feature_vector(selected_symptoms, all_symptoms):
    vec = np.zeros(len(all_symptoms), dtype=np.float32)
    sym_lower = [s.lower().strip() for s in all_symptoms]
    for sym in selected_symptoms:
        key = sym.lower().strip()
        if key in sym_lower:
            vec[sym_lower.index(key)] = 1.0
    return vec.reshape(1, -1)


def predict_diseases(selected_symptoms, artifacts, top_k=10):
    all_symptoms   = artifacts["symptoms"]
    label_encoder  = artifacts["label_encoder"]
    dnn_model      = artifacts["dnn_model"]
    et_model       = artifacts["et_model"]

    if all_symptoms is None or label_encoder is None:
        return None, "Artifacts not loaded correctly."

    X = build_feature_vector(selected_symptoms, all_symptoms)

    if dnn_model is not None:
        probs = dnn_model.predict(X, verbose=0)[0]
    elif et_model is not None:
        probs = et_model.predict_proba(X)[0]
    else:
        return None, "No model available."

    top_idx  = np.argsort(probs)[::-1][:top_k]
    top_prob = probs[top_idx]

    try:
        top_names = label_encoder.inverse_transform(top_idx)
    except Exception:
        top_names = [f"Disease_{i}" for i in top_idx]

    results = [
        {"rank": i + 1, "disease": name, "probability": float(prob)}
        for i, (name, prob) in enumerate(zip(top_names, top_prob))
    ]
    return results, None


# ─────────────────────────────────────────────
# Emergency Scoring
# ─────────────────────────────────────────────
def compute_emergency(selected_symptoms):
    selected_lower = [s.lower().strip() for s in selected_symptoms]
    score = 0
    triggered = []
    for sym, weight in EMERGENCY_WEIGHTS.items():
        if sym.lower() in selected_lower:
            score += weight
            triggered.append((sym, weight))
    if score >= 5:
        level = "CRITICAL"
    elif score >= 3:
        level = "HIGH"
    else:
        level = "NORMAL"
    return level, score, triggered


# ─────────────────────────────────────────────
# Confidence Category
# ─────────────────────────────────────────────
def confidence_category(prob):
    pct = prob * 100
    if pct >= CONFIDENCE_THRESHOLDS["Very High"]:
        return "Very High", "conf-very-high"
    elif pct >= CONFIDENCE_THRESHOLDS["High"]:
        return "High", "conf-high"
    elif pct >= CONFIDENCE_THRESHOLDS["Moderate"]:
        return "Moderate", "conf-moderate"
    else:
        return "Low", "conf-low"


# ─────────────────────────────────────────────
# Clinical Recommendation
# ─────────────────────────────────────────────
def generate_clinical_text(disease, confidence_pct, conf_cat, emergency_level, selected_symptoms):
    emergency_advice = {
        "CRITICAL": (
            "**IMMEDIATE ACTION REQUIRED.** The selected symptoms include one or more critical emergency indicators. "
            "Call emergency services (112 / 102) immediately or proceed to the nearest emergency department without delay. "
            "Do not attempt self-medication."
        ),
        "HIGH": (
            "**Urgent medical attention is advised.** The symptom pattern suggests elevated risk. "
            "Visit a physician or urgent care clinic within the next few hours. "
            "Monitor for symptom escalation and avoid strenuous activity."
        ),
        "NORMAL": (
            "Schedule a consultation with a qualified physician at the earliest convenience. "
            "Maintain a symptom diary for accurate clinical documentation. "
            "Do not use this AI assessment as a substitute for professional diagnosis."
        ),
    }

    sym_list = ", ".join(selected_symptoms[:8]) + ("..." if len(selected_symptoms) > 8 else "")

    text = f"""
**Predicted Condition:** {disease}
**AI Confidence:** {confidence_pct:.1f}% ({conf_cat})
**Emergency Risk Level:** {emergency_level}
**Reported Symptoms ({len(selected_symptoms)}):** {sym_list}

---

**Clinical Recommendation:**
{emergency_advice[emergency_level]}

**General Guidance:**
- Provide your complete symptom history, onset timeline, and medical records to your physician.
- Avoid self-diagnosis or self-medication solely based on this AI output.
- If symptoms worsen suddenly, escalate care immediately regardless of current risk level.

**Disclaimer:** HealthBandhu is a research-grade clinical decision-support tool. It is not a substitute for professional medical advice, diagnosis, or treatment.
"""
    return text.strip()


# ─────────────────────────────────────────────
# PDF Report Generator  (FIXED)
# ─────────────────────────────────────────────
def generate_report(disease, confidence_pct, conf_cat, emergency_level, emergency_score,
                    triggered_symptoms, selected_symptoms, top_results):
    """
    Generate a clinical PDF report.

    Uses fpdf2 if installed (rich, formatted PDF).
    Falls back to a clean stdlib-only PDF with correct absolute text positioning.

    Key fixes vs original:
      - fpdf2 path: removed the premature pdf.output() call that was outputting
        an empty PDF mid-setup and discarding all subsequent content.
      - fpdf2 path: removed the duplicate hardcoded table header row that caused
        rank-1 disease to appear twice in the differential diagnoses table.
      - fpdf2 path: replaced deprecated ln=True parameter with
        new_x="LMARGIN", new_y="NEXT" for fpdf2 v2.x compatibility.
      - fpdf2 path: corrected set_margins / set_auto_page_break ordering so
        they are set once before add_page(), not split across the output call.
      - stdlib fallback: replaced the broken relative Td operator (which stacked
        offsets and caused all text to overlap) with the Tm absolute-positioning
        operator, and added proper multi-page support.
      - Both paths: removed bullet character (U+2022) which is outside the
        latin-1 range supported by Helvetica core font; replaced with '-'.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ──────────────────────────────────────────────────────────────
    # PATH A: Rich PDF via fpdf2
    # ──────────────────────────────────────────────────────────────
    if _USE_FPDF:
        from fpdf import FPDF

        class HBandhuPDF(FPDF):
            def header(self):
                self.set_fill_color(10, 37, 64)
                self.rect(0, 0, 210, 30, "F")
                self.set_xy(0, 6)
                self.set_font("Helvetica", "B", 16)
                self.set_text_color(255, 255, 255)
                self.cell(
                    0, 8,
                    "HealthBandhu - AI Clinical Diagnostic Report",
                    align="C",
                    new_x="LMARGIN", new_y="NEXT",
                )
                self.set_font("Helvetica", "", 8)
                self.set_text_color(220, 230, 240)
                self.cell(
                    0, 5,
                    "Explainable Hybrid AI  |  UEM Kolkata  |  Clinical Decision Support",
                    align="C",
                    new_x="LMARGIN", new_y="NEXT",
                )
                self.ln(6)
                self.set_text_color(30, 30, 30)

            def footer(self):
                self.set_y(-13)
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(150, 150, 150)
                self.cell(
                    0, 8,
                    f"Page {self.page_no()}  |  HealthBandhu  |  "
                    f"Joy Ghatak, B.Tech CSE, UEM Kolkata  |  {now}",
                    align="C",
                )

            def section_title(self, title, r=10, g=37, b=64):
                self.set_font("Helvetica", "B", 9)
                self.set_fill_color(r, g, b)
                self.set_text_color(255, 255, 255)
                self.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(30, 30, 30)
                self.ln(2)

            def kv(self, key, value, shade=False):
                KEY_W, VAL_W = 58, 125
                if shade:
                    self.set_fill_color(241, 245, 249)
                self.set_font("Helvetica", "B", 9)
                self.cell(KEY_W, 6, f"  {key}", fill=shade)
                self.set_font("Helvetica", "", 9)
                self.cell(VAL_W, 6, str(value)[:70], fill=shade, new_x="LMARGIN", new_y="NEXT")

        # Create PDF – set layout BEFORE add_page() so header renders correctly
        pdf = HBandhuPDF()
        pdf.set_margins(15, 38, 15)
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()

        # ── Report Information ──────────────────────────────────
        pdf.section_title("REPORT INFORMATION")
        pdf.kv("Generated",   now,                                        shade=True)
        pdf.kv("System",      "HealthBandhu v1.0  |  UEM Kolkata")
        pdf.kv("Developer",   "Joy Ghatak, B.Tech CSE",                   shade=True)
        pdf.kv("Model",       "DNN (TensorFlow) + Extra Trees Classifier")
        pdf.ln(4)

        # ── Primary Diagnosis ──────────────────────────────────
        pdf.section_title("PRIMARY DIAGNOSIS", r=26, g=111, b=168)
        pdf.kv("Predicted Disease", disease,                   shade=True)
        pdf.kv("AI Confidence",     f"{confidence_pct:.2f}%")
        pdf.kv("Confidence Level",  conf_cat,                  shade=True)
        pdf.ln(4)

        # ── Emergency Risk ─────────────────────────────────────
        er_colors = {
            "CRITICAL": (220, 38,  38),
            "HIGH":     (245, 158, 11),
            "NORMAL":   (22,  163, 74),
        }
        r2, g2, b2 = er_colors.get(emergency_level, (10, 37, 64))
        pdf.section_title(f"EMERGENCY RISK  -  {emergency_level}", r=r2, g=g2, b=b2)
        pdf.kv("Risk Level",     emergency_level,   shade=True)
        pdf.kv("Weighted Score", str(emergency_score))
        if triggered_symptoms:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "  Critical Symptoms:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for sym, w in triggered_symptoms:
                pdf.cell(
                    0, 5,
                    f"     ! {sym.title()}  (severity weight: {w})",
                    new_x="LMARGIN", new_y="NEXT",
                )
        else:
            pdf.kv("Status", "No critical emergency symptoms detected.", shade=True)
        pdf.ln(4)

        # ── Symptoms Reported ──────────────────────────────────
        pdf.section_title("SYMPTOMS REPORTED", r=13, g=148, b=136)
        pdf.kv("Total Symptoms", str(len(selected_symptoms)), shade=True)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 8.5)
        COL = 90
        for i in range(0, len(selected_symptoms), 2):
            left_sym  = f"  - {selected_symptoms[i]}"
            right_sym = f"  - {selected_symptoms[i + 1]}" if i + 1 < len(selected_symptoms) else ""
            pdf.cell(COL, 5, left_sym[:46])
            pdf.cell(COL, 5, right_sym[:46], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ── Top-10 Differential Diagnoses ─────────────────────
        pdf.section_title("TOP-10 DIFFERENTIAL DIAGNOSES", r=26, g=111, b=168)

        # Table header row
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_fill_color(10, 37, 64)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(14,  8, "Rank",        border=1, fill=True)
        pdf.cell(138, 8, "Disease",     border=1, fill=True)
        pdf.cell(0,   8, "Probability", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(30, 30, 30)

        # Table data rows  (no duplicate header — original bug fixed here)
        for i, res in enumerate(top_results):
            shade = (i % 2 == 0)
            if shade:
                pdf.set_fill_color(241, 245, 249)
            pdf.set_font("Helvetica", "B" if i == 0 else "", 8.5)
            pdf.cell(14,  6, str(res["rank"]),                fill=shade, border=1)
            pdf.cell(138, 6, str(res["disease"])[:58],        fill=shade, border=1)
            pdf.cell(0,   6, f"{res['probability']*100:.2f}%", fill=shade, border=1,
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ── Clinical Recommendation ────────────────────────────
        pdf.section_title("CLINICAL RECOMMENDATION", r=13, g=148, b=136)
        advice = {
            "CRITICAL": (
                "IMMEDIATE ACTION REQUIRED. Call emergency services (112) immediately "
                "or go to the nearest emergency department without delay. "
                "Do not attempt self-medication."
            ),
            "HIGH": (
                "Urgent medical attention is advised. Visit a physician or urgent care "
                "clinic within the next few hours. Monitor for symptom escalation and "
                "avoid strenuous activity."
            ),
            "NORMAL": (
                "Schedule a consultation with a qualified physician at the earliest "
                "convenience. Maintain a symptom diary for accurate clinical documentation."
            ),
        }
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 6, advice.get(emergency_level, ""))
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(0, 5, "  General Guidance:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8.5)
        for line in [
            "1. Share your complete symptom history with your physician.",
            "2. Bring prior medical records to your consultation.",
            "3. Do not self-medicate based solely on this AI output.",
            "4. If symptoms worsen suddenly, seek emergency care immediately.",
        ]:
            pdf.cell(0, 5, f"     {line}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ── Medical Disclaimer ─────────────────────────────────
        pdf.section_title("MEDICAL DISCLAIMER", r=127, g=29, b=29)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(127, 29, 29)
        for line in [
            "This report is generated by HealthBandhu, an AI research system.",
            "It is NOT a substitute for professional medical diagnosis or treatment.",
            "HealthBandhu is NOT a licensed medical device.",
            "Always consult a qualified healthcare provider.",
            "In case of emergency, call 112 immediately.",
        ]:
            pdf.cell(0, 5, f"  {line}", new_x="LMARGIN", new_y="NEXT")

        # Return bytes — call output() ONCE at the very end
        return bytes(pdf.output())

    # ──────────────────────────────────────────────────────────────
    # PATH B: Minimal stdlib-only PDF (no extra packages required)
    #
    # Fix: uses Tm (absolute text matrix) instead of Td (relative
    # offset), so each line is positioned independently and never
    # overlaps.  Also supports multiple pages correctly.
    # ──────────────────────────────────────────────────────────────
    lines = [
        "HealthBandhu - AI Clinical Diagnostic Report",
        "=" * 60,
        f"Generated       : {now}",
        f"Developer       : Joy Ghatak, B.Tech CSE, UEM Kolkata",
        f"System          : HealthBandhu v1.0",
        f"Model           : DNN (TensorFlow) + Extra Trees Classifier",
        "",
        "PRIMARY DIAGNOSIS",
        "-" * 60,
        f"Predicted Disease : {disease}",
        f"AI Confidence     : {confidence_pct:.2f}%",
        f"Confidence Level  : {conf_cat}",
        "",
        f"EMERGENCY RISK ASSESSMENT  -  {emergency_level}",
        "-" * 60,
        f"Risk Level        : {emergency_level}",
        f"Weighted Score    : {emergency_score}",
    ]
    if triggered_symptoms:
        lines.append("Critical Symptoms :")
        for sym, w in triggered_symptoms:
            lines.append(f"  ! {sym.title()} (weight: {w})")
    else:
        lines.append("Status            : No critical emergency symptoms detected.")

    lines += [
        "", "SYMPTOMS REPORTED", "-" * 60,
        f"Total : {len(selected_symptoms)}",
    ]
    for i, sym in enumerate(selected_symptoms, 1):
        lines.append(f"  {i:2d}. {sym}")

    lines += [
        "", "TOP-10 DIFFERENTIAL DIAGNOSES", "-" * 60,
        f"{'Rank':<6}{'Disease':<50}{'Probability':>12}",
    ]
    for res in top_results:
        lines.append(
            f"{res['rank']:<6}{str(res['disease'])[:49]:<50}"
            f"{res['probability'] * 100:>11.2f}%"
        )

    advice_map = {
        "CRITICAL": "IMMEDIATE: Call 112 / go to emergency department NOW.",
        "HIGH":     "URGENT: Visit a physician within the next few hours.",
        "NORMAL":   "Schedule a consultation with a qualified physician.",
    }
    lines += [
        "", "CLINICAL RECOMMENDATION", "-" * 60,
        advice_map.get(emergency_level, ""),
        "", "General Guidance:",
        "1. Share your complete symptom history with your physician.",
        "2. Bring prior medical records to your consultation.",
        "3. Do not self-medicate based solely on this AI output.",
        "4. If symptoms worsen suddenly, seek emergency care immediately.",
        "",
        "MEDICAL DISCLAIMER", "-" * 60,
        "This report is generated by HealthBandhu, an AI research system.",
        "It is NOT a substitute for professional medical diagnosis or treatment.",
        "HealthBandhu is NOT a licensed medical device.",
        "Always consult a qualified healthcare provider.",
        "In case of emergency, call 112 immediately.",
    ]

    # ── Page layout constants ──
    PAGE_W      = 595
    PAGE_H      = 842
    MARGIN_L    = 50
    MARGIN_TOP  = PAGE_H - 50   # y coordinate of first text line (PDF origin = bottom-left)
    MARGIN_BOT  = 50            # y coordinate below which we start a new page
    LINE_H      = 14            # points per line

    def _safe(s):
        """Truncate, encode to latin-1 safely, escape PDF special chars."""
        s = str(s)[:95]
        s = s.encode("latin-1", errors="replace").decode("latin-1")
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # ── Build page content streams ──
    # Each page stream is built independently using Tm for absolute positioning.
    pages_streams = []
    ops = []
    cur_y = MARGIN_TOP

    def _flush_page():
        stream = "BT\n/F1 10 Tf\n" + "".join(ops) + "ET\n"
        pages_streams.append(stream.encode("latin-1", errors="replace"))
        ops.clear()

    for raw_line in lines:
        # Start new page if we've run out of vertical space
        if cur_y < MARGIN_BOT:
            _flush_page()
            cur_y = MARGIN_TOP

        # Tm sets the absolute text matrix: 1 0 0 1 x y Tm
        ops.append(f"1 0 0 1 {MARGIN_L} {cur_y:.1f} Tm\n")
        ops.append(f"({_safe(raw_line)}) Tj\n")
        cur_y -= LINE_H

    if ops:
        _flush_page()

    n_pages = len(pages_streams)

    # ── Assemble PDF object tree ──
    # Object numbering:
    #   1          = Catalog
    #   2          = Pages dict
    #   3 … 3+P-1  = Page objects      (P = n_pages)
    #   3+P … 3+2P-1 = Content streams
    #   3+2P       = Font resource

    OBJ_CATALOG   = 1
    OBJ_PAGES     = 2
    OBJ_PAGE_0    = 3
    OBJ_CONTENT_0 = 3 + n_pages
    OBJ_FONT      = 3 + 2 * n_pages

    body    = io.BytesIO()
    xref    = []   # byte offsets for each object (index = obj_num - 1)

    body.write(b"%PDF-1.4\n")

    def _write_obj(num, header, stream_data=None):
        """Write a PDF object; stream_data is raw bytes for stream objects."""
        xref.append((num, body.tell()))
        body.write(f"{num} 0 obj\n".encode())
        if stream_data is not None:
            body.write(f"{header}/Length {len(stream_data)} >>\nstream\n".encode())
            body.write(stream_data)
            body.write(b"\nendstream\n")
        else:
            body.write(header.encode())
        body.write(b"endobj\n")

    # Object 1: Catalog
    _write_obj(OBJ_CATALOG,
               f"<< /Type /Catalog /Pages {OBJ_PAGES} 0 R >>\n")

    # Object 2: Pages dict
    kids_str = " ".join(f"{OBJ_PAGE_0 + i} 0 R" for i in range(n_pages))
    _write_obj(OBJ_PAGES,
               f"<< /Type /Pages /Kids [{kids_str}] /Count {n_pages} >>\n")

    # Page objects
    for i in range(n_pages):
        _write_obj(
            OBJ_PAGE_0 + i,
            f"<< /Type /Page /Parent {OBJ_PAGES} 0 R "
            f"/MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Contents {OBJ_CONTENT_0 + i} 0 R "
            f"/Resources << /Font << /F1 {OBJ_FONT} 0 R >> >> >>\n",
        )

    # Content stream objects
    for i, stream_bytes in enumerate(pages_streams):
        _write_obj(OBJ_CONTENT_0 + i, "<< ", stream_bytes)

    # Font object
    _write_obj(
        OBJ_FONT,
        "<< /Type /Font /Subtype /Type1 "
        "/BaseFont /Helvetica "
        "/Encoding /WinAnsiEncoding >>\n",
    )

    # ── Cross-reference table ──
    total_objs = OBJ_FONT + 1          # objects numbered 1 … OBJ_FONT
    xref_offset = body.tell()

    # Sort by object number to build the xref table in order
    xref_sorted = sorted(xref, key=lambda t: t[0])
    offsets_in_order = [off for _, off in xref_sorted]

    body.write(f"xref\n0 {total_objs}\n".encode())
    body.write(b"0000000000 65535 f \n")   # free-list entry for obj 0
    for off in offsets_in_order:
        body.write(f"{off:010d} 00000 n \n".encode())

    body.write(
        f"trailer\n<< /Size {total_objs} /Root {OBJ_CATALOG} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )

    return body.getvalue()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar(artifacts):
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div style="font-size:2rem;margin-bottom:6px;">🏥</div>
            <h2>HealthBandhu</h2>
            <p>Explainable Hybrid AI Framework for<br>Multiclass Disease Diagnosis &amp;<br>Emergency Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <h4>📊 Dataset Statistics</h4>
            <div class="sidebar-stat"><span>Patient Records</span><span class="val">246,945</span></div>
            <div class="sidebar-stat"><span>Symptom Features</span><span class="val">377</span></div>
            <div class="sidebar-stat"><span>Disease Classes</span><span class="val">754</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <h4>🎯 Model Performance</h4>
            <div class="sidebar-stat"><span>Top-1 Accuracy</span><span class="val">85.70%</span></div>
            <div class="sidebar-stat"><span>Top-3 Accuracy</span><span class="val">95.87%</span></div>
            <div class="sidebar-stat"><span>Top-5 Accuracy</span><span class="val">97.92%</span></div>
            <div class="sidebar-stat"><span>Top-10 Accuracy</span><span class="val">99.35%</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <h4>🤖 AI Components</h4>
            <div class="sidebar-stat"><span>Primary Model</span><span class="val">DNN</span></div>
            <div class="sidebar-stat"><span>Secondary Model</span><span class="val">Extra Trees</span></div>
            <div class="sidebar-stat"><span>Explainability</span><span class="val">XAI Layer</span></div>
            <div class="sidebar-stat"><span>Emergency Engine</span><span class="val">Rule-Based</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <h4>👤 Developer</h4>
            <div class="sidebar-stat"><span>Name</span><span class="val">Joy Ghatak</span></div>
            <div class="sidebar-stat"><span>Degree</span><span class="val">B.Tech CSE</span></div>
            <div class="sidebar-stat"><span>Institute</span><span class="val">UEM Kolkata</span></div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    with st.spinner("Loading HealthBandhu AI models..."):
        artifacts, errors = load_all_artifacts()

    render_sidebar(artifacts)

    # ── Top Header ──
    st.markdown("""
    <div class="top-header">
        <div class="top-header-icon">🏥</div>
        <div class="top-header-text">
            <h1>HealthBandhu</h1>
            <p>Explainable Hybrid AI Framework for Multiclass Disease Diagnosis,
               Emergency Risk Assessment &amp; Clinical Decision Support</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if errors:
        for err in errors:
            st.warning(f"⚠️ {err}", icon=None)

    all_symptoms = artifacts.get("symptoms")
    if all_symptoms is None:
        st.error("❌ **healthbandhu_symptoms.pkl** could not be loaded. Cannot proceed with symptom selection.")
        st.stop()

    all_symptoms_list = list(all_symptoms)

    # ─────────────────────────────────────────
    # SECTION 1 – Symptom Selection
    # ─────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="icon">🔍</span>
        <h2>Symptom Selection</h2>
        <span class="tag">Step 1</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Search and select all symptoms the patient is currently experiencing.
        The AI engine supports all <strong>377 clinical symptoms</strong> from the HealthBandhu dataset.
        Select at least one symptom to generate a diagnosis.
    </div>
    """, unsafe_allow_html=True)

    col_search, col_clear = st.columns([5, 1])
    with col_search:
        search_query = st.text_input(
            "Search symptoms",
            placeholder="Type to filter symptoms (e.g. 'headache', 'chest pain')...",
            label_visibility="collapsed",
        )
    with col_clear:
        clear_btn = st.button("Clear All", key="clear_btn")

    if clear_btn:
        st.session_state["selected_symptoms"] = []
        st.rerun()

    filtered = (
        [s for s in all_symptoms_list if search_query.lower() in s.lower()]
        if search_query.strip()
        else all_symptoms_list
    )

    if "selected_symptoms" not in st.session_state:
        st.session_state["selected_symptoms"] = []

    selected_symptoms = st.multiselect(
        f"Select Symptoms ({len(filtered)} available)",
        options=filtered,
        default=[s for s in st.session_state["selected_symptoms"] if s in filtered],
        placeholder="Choose symptoms from the list...",
        key="symptom_multiselect",
    )
    st.session_state["selected_symptoms"] = selected_symptoms

    if selected_symptoms:
        st.markdown(f"**{len(selected_symptoms)} symptom(s) selected.**")

    st.markdown("<hr class='hb-divider'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # SECTION 2 – Run Diagnosis
    # ─────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="icon">🧠</span>
        <h2>AI Diagnosis Engine</h2>
        <span class="tag">Step 2</span>
    </div>
    """, unsafe_allow_html=True)

    run_col, _ = st.columns([2, 5])
    with run_col:
        run_btn = st.button("▶  Run Diagnosis", key="run_btn")

    if run_btn:
        if not selected_symptoms:
            st.error("⚠️  Please select at least one symptom before running the diagnosis.")
        else:
            with st.spinner("Analysing symptom pattern with HealthBandhu AI..."):
                top_results, err = predict_diseases(selected_symptoms, artifacts, top_k=10)

            if err:
                st.error(f"❌ Prediction failed: {err}")
            else:
                best      = top_results[0]
                disease   = best["disease"]
                prob      = best["probability"]
                conf_pct  = prob * 100
                conf_cat, conf_cls = confidence_category(prob)

                st.markdown(f"""
                <div class="result-main">
                    <div style="font-size:0.72rem;font-weight:700;color:#1A6FA8;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:8px;">Primary Diagnosis</div>
                    <div class="disease-name">{disease}</div>
                    <div class="confidence-row">
                        <div style="font-size:1.4rem;font-weight:700;color:#0A2540;
                                    font-family:'JetBrains Mono',monospace;">
                            {conf_pct:.1f}%
                        </div>
                        <span class="conf-badge {conf_cls}">{conf_cat} Confidence</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Emergency Detection ──
                st.markdown("""
                <div class="section-header" style="margin-top:28px;">
                    <span class="icon">🚨</span>
                    <h2>Emergency Risk Assessment</h2>
                    <span class="tag">Real-time</span>
                </div>
                """, unsafe_allow_html=True)

                emerg_level, emerg_score, triggered = compute_emergency(selected_symptoms)

                if emerg_level == "CRITICAL":
                    st.markdown(f"""
                    <div class="emergency-critical">
                        <div class="emergency-title" style="color:#DC2626;">🚨 CRITICAL EMERGENCY RISK</div>
                        <div class="emergency-body" style="color:#7F1D1D;">
                            <strong>Weighted Emergency Score: {emerg_score}</strong><br>
                            One or more life-threatening symptoms detected.
                            <strong>Call emergency services (112 / 102) immediately.</strong>
                            Proceed to the nearest emergency department without delay.<br><br>
                            <strong>Critical symptoms identified:</strong><br>
                            {"<br>".join([f"- {s.title()} (severity weight: {w})" for s, w in triggered])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                elif emerg_level == "HIGH":
                    st.markdown(f"""
                    <div class="emergency-high">
                        <div class="emergency-title" style="color:#B45309;">⚠️ HIGH EMERGENCY RISK</div>
                        <div class="emergency-body" style="color:#78350F;">
                            <strong>Weighted Emergency Score: {emerg_score}</strong><br>
                            Elevated risk indicators detected. Seek urgent medical attention within the next few hours.<br><br>
                            <strong>Warning symptoms identified:</strong><br>
                            {"<br>".join([f"- {s.title()} (severity weight: {w})" for s, w in triggered])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                    <div class="emergency-normal">
                        <div class="emergency-title" style="color:#16A34A;">✅ NORMAL RISK</div>
                        <div class="emergency-body" style="color:#14532D;">
                            <strong>Weighted Emergency Score: {emerg_score}</strong><br>
                            No critical emergency symptoms detected at this time.
                            Schedule a consultation with your physician at the earliest convenience.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<hr class='hb-divider'>", unsafe_allow_html=True)

                # ── Top-10 Predictions ──
                st.markdown("""
                <div class="section-header">
                    <span class="icon">📊</span>
                    <h2>Top-10 Differential Diagnoses</h2>
                    <span class="tag">AI Output</span>
                </div>
                """, unsafe_allow_html=True)

                df = pd.DataFrame([
                    {
                        "Rank":        r["rank"],
                        "Disease":     r["disease"],
                        "Probability": f"{r['probability']*100:.2f}%",
                        "Confidence":  confidence_category(r["probability"])[0],
                    }
                    for r in top_results
                ])

                tab_table, tab_chart = st.tabs(["📋  Table View", "📈  Chart View"])

                with tab_table:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                with tab_chart:
                    chart_df = pd.DataFrame({
                        "Disease":     [r["disease"][:40] for r in top_results],
                        "Probability": [round(r["probability"] * 100, 2) for r in top_results],
                    }).sort_values("Probability")
                    st.bar_chart(chart_df.set_index("Disease"), color="#1A6FA8", height=380)

                st.markdown("<hr class='hb-divider'>", unsafe_allow_html=True)

                # ── Clinical Assistant ──
                st.markdown("""
                <div class="section-header">
                    <span class="icon">💬</span>
                    <h2>Clinical Assistant</h2>
                    <span class="tag">Decision Support</span>
                </div>
                """, unsafe_allow_html=True)

                clinical_text = generate_clinical_text(
                    disease, conf_pct, conf_cat, emerg_level, selected_symptoms
                )

                st.markdown(f"""
                <div class="clinical-box">
                    <h4>🩺 Clinical Decision Support Output</h4>
                    {clinical_text.replace(chr(10), "<br>")}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<hr class='hb-divider'>", unsafe_allow_html=True)

                # ── Report Generator ──
                st.markdown("""
                <div class="section-header">
                    <span class="icon">📄</span>
                    <h2>Report Generator</h2>
                    <span class="tag">Download</span>
                </div>
                """, unsafe_allow_html=True)

                try:
                    pdf_bytes = generate_report(
                        disease, conf_pct, conf_cat, emerg_level, emerg_score,
                        triggered, selected_symptoms, top_results,
                    )
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="⬇  Download Clinical Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"HealthBandhu_Report_{timestamp}.pdf",
                        mime="application/pdf",
                    )
                except Exception as _pdf_err:
                    st.warning(
                        f"PDF generation error: {_pdf_err}. "
                        "Install fpdf2 (`pip install fpdf2`) to enable rich PDF reports."
                    )

                st.markdown("""
                <div class="info-box">
                    The generated report contains the full clinical assessment including disease prediction,
                    confidence scores, emergency risk evaluation, and evidence-based recommendations.
                    Share this report with your physician.
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # SECTION 3 – Performance Dashboard
    # ─────────────────────────────────────────
    st.markdown("<hr class='hb-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="icon">📈</span>
        <h2>Model Performance Dashboard</h2>
        <span class="tag">Metrics</span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (label, val) in zip([m1, m2, m3, m4], MODEL_METRICS.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val:.2f}%</div>
                <div class="sub">246,945 patient records</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    info_left, info_right = st.columns(2)
    with info_left:
        st.markdown("""
        <div class="clinical-box">
            <h4>🏗 Model Architecture</h4>
            <strong>Primary:</strong> Deep Neural Network (DNN) – .keras format<br>
            <strong>Secondary:</strong> Extra Trees Classifier – ensemble fallback<br>
            <strong>Input:</strong> 377-dimensional binary symptom vector<br>
            <strong>Output:</strong> Softmax probability over 754 disease classes<br>
            <strong>Explainability:</strong> Top-K prediction ranking + confidence scoring<br>
            <strong>Emergency Layer:</strong> Weighted rule-based critical symptom engine
        </div>
        """, unsafe_allow_html=True)
    with info_right:
        st.markdown("""
        <div class="clinical-box">
            <h4>📋 Dataset Information</h4>
            <strong>Source:</strong> Curated multiclass clinical symptom dataset<br>
            <strong>Records:</strong> 246,945 patient entries<br>
            <strong>Features:</strong> 377 binary symptom flags<br>
            <strong>Classes:</strong> 754 distinct disease categories<br>
            <strong>Validation:</strong> Stratified K-Fold cross-validation<br>
            <strong>Format:</strong> IEEE-ready benchmark evaluation
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # SECTION 4 – Architecture Diagram
    # ─────────────────────────────────────────
    st.markdown("<hr class='hb-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="icon">🏛</span>
        <h2>HealthBandhu System Architecture</h2>
        <span class="tag">Framework</span>
    </div>
    """, unsafe_allow_html=True)

    arch_steps = [
        ("Symptom Input Layer",         "377 binary symptom features from patient report",            "🩺"),
        ("Emergency Rule Engine",        "Weighted critical symptom scoring – threshold-based triage", "🚨"),
        ("Feature Processing",           "Binary vector encoding aligned to training vocabulary",       "⚙️"),
        ("Deep Neural Network",          "Primary DNN model (healthbandhu_dnn.keras) – 754 classes",   "🧠"),
        ("Explainable AI Layer",         "Top-K ranking, confidence scoring, probability calibration", "🔍"),
        ("Clinical Decision Assistant",  "Natural language recommendation and risk communication",      "💬"),
        ("Medical Report Generator",     "Downloadable structured clinical report (PDF)",              "📄"),
    ]

    st.markdown('<div class="arch-box"><h4>⚕ HealthBandhu Pipeline Architecture</h4>', unsafe_allow_html=True)
    for i, (label, desc, icon) in enumerate(arch_steps):
        arrow = "" if i == len(arch_steps) - 1 else "↓"
        st.markdown(f"""
        <div class="arch-step">
            <div class="arch-num">{i+1}</div>
            <div>
                <div class="arch-label">{icon} {label}</div>
                <div class="arch-desc">{desc}</div>
            </div>
            <div class="arch-arrow">{arrow}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # Footer
    # ─────────────────────────────────────────
    st.markdown("<hr class='hb-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:20px 0 8px 0;color:#94A3B8;font-size:0.75rem;line-height:2;">
        <strong style="color:#1A6FA8;">HealthBandhu</strong> &nbsp;·&nbsp;
        Explainable Hybrid AI Framework for Clinical Decision Support<br>
        Developed by <strong>Joy Ghatak</strong> | B.Tech CSE | IEM Newtown | UEM Kolkata<br>
        <span style="color:#CBD5E1;">⚕ Not a substitute for professional medical advice ⚕</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()