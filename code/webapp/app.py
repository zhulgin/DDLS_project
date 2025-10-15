# app.py — Streamlit web app for calibrated H-NMR classification (robust load + logs)

import os, io, json, sys, math, re
# ---- avoid GPU/Metal lockups & quiet TF logs ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- local imports ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from utils.calibration import CalibratedPredictor  # uses temperature.json

# ---------- path resolver ----------
APP_DIR  = Path(__file__).resolve().parent          # .../code/webapp
CODE_DIR = APP_DIR.parent                           # .../code
PROJ_DIR = CODE_DIR.parent                          # project root
CANDIDATES = [PROJ_DIR / "models", CODE_DIR / "models"]
MODELS_DIR = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])

BASELINE_DIR = MODELS_DIR / "baseline"
CALIB_DIR    = MODELS_DIR / "calibrated_v1"
MODEL_PATH  = BASELINE_DIR / "model.keras"
BINS_PATH   = BASELINE_DIR / "bins.npy"
CFG_PATH    = BASELINE_DIR / "config.json"
LABELS_PATH = BASELINE_DIR / "class_order.json"     # optional
TEMP_PATH   = CALIB_DIR   / "temperature.json"

# ---------- lazy TF import ----------
_tf = None
def get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        # try to ensure CPU only
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        _tf = tf
    return _tf

# ---------- artifacts (loaded after file parsed) ----------
@st.cache_resource(show_spinner=False)
def load_artifacts_with_logs():
    """Load model/bins/config with detailed progress messages."""
    logs = []
    def log(msg): logs.append(msg)

    log("Importing TensorFlow…")
    tf = get_tf()
    log(f"TF version: {tf.__version__}")

    # Load model (be liberal: compile=False; try keras then tf.keras)
    log(f"Loading model: {MODEL_PATH}")
    model = None
    load_err = None
    try:
        from keras.models import load_model as k_load_model  # Keras 3 loader
        model = k_load_model(str(MODEL_PATH), compile=False)
        log("Loaded model via keras.models.load_model.")
    except Exception as e1:
        load_err = e1
        try:
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            log("Loaded model via tf.keras.models.load_model.")
            load_err = None
        except Exception as e2:
            load_err = (e1, e2)

    if load_err is not None:
        raise RuntimeError(f"Failed to load model: {load_err}")

    # Bins / config / labels
    log(f"Loading bins: {BINS_PATH}")
    bins = np.load(str(BINS_PATH))

    log(f"Loading config: {CFG_PATH}")
    cfg  = json.load(open(CFG_PATH))

    labels = None
    if LABELS_PATH.exists():
        try:
            labels = json.load(open(LABELS_PATH))
            log(f"Loaded labels: {LABELS_PATH}")
        except Exception as e:
            log(f"Labels load failed (continuing without): {e}")

    log(f"Loading temperature: {TEMP_PATH}")
    predictor = CalibratedPredictor(model, str(TEMP_PATH))

    # Extra: report model input shape
    try:
        ishape = model.input_shape
        log(f"model.input_shape = {ishape}")
    except Exception:
        log("model.input_shape = (unknown)")

    return predictor, bins, cfg, labels, logs

# ---------- parsing ----------
def _parse_two_column(txt: str) -> pd.DataFrame:
    """
    Parse plain peak lists with two columns (ppm, intensity) — CSV/TSV/whitespace.
    If a header row like 'ppm,intensity' exists, it's ignored.
    """
    first = (txt.splitlines() + [""])[0]
    delim = "," if "," in first else ("\t" if "\t" in first else None)
    try:
        df = pd.read_csv(io.StringIO(txt), sep=delim, header=None, comment="#")
    except Exception:
        df = pd.read_csv(io.StringIO(txt), sep=r"\s+", header=None, comment="#", engine="python")

    # drop any header-looking first row
    if df.shape[1] >= 2 and str(df.iloc[0,0]).strip().lower().startswith("ppm"):
        df = df.iloc[1:]

    if df.shape[1] < 2:
        raise ValueError("Expected at least two columns with ppm and intensity.")
    df = df.iloc[:, :2]
    df.columns = ["ppm", "intensity"]

    # coerce numerics
    df["ppm"] = pd.to_numeric(df["ppm"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    return df.dropna().astype(float)

def _parse_hmdb_table_of_peaks(txt: str) -> pd.DataFrame:
    """
    Parse HMDB TXT with 'Table of Peaks' section.
    We extract only (ppm, Height). Multiplets/Assignments sections are ignored.
    """
    lines = txt.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.search(r"^\s*Table of Peaks\s*$", line, re.IGNORECASE):
            start = i
            break
    if start is None:
        raise ValueError("No 'Table of Peaks' section found.")

    # find the column header line that contains 'ppm' and 'Height'
    i = start + 1
    while i < len(lines) and not (("ppm" in lines[i]) and ("Height" in lines[i])):
        i += 1
    if i >= len(lines):
        raise ValueError("Could not locate peaks column header.")

    # move to first data row
    i += 1

    # Data pattern: "<No.>  <ppm>  <Height>  ..."
    pat = re.compile(r"^\s*\d+\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)")
    rows = []
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.lower().startswith("table of"):
            break
        m = pat.match(s)
        if m:
            ppm = float(m.group(1))
            height = float(m.group(2))
            rows.append((ppm, height))
        i += 1

    if not rows:
        raise ValueError("No peak rows parsed from HMDB 'Table of Peaks'.")
    return pd.DataFrame(rows, columns=["ppm", "intensity"])

def parse_bytes(raw: bytes) -> pd.DataFrame:
    """
    Robust parser: try HMDB 'Table of Peaks' first; fall back to generic two-column peaks.
    """
    txt = raw.decode("utf-8", errors="ignore")
    if "Table of Peaks" in txt:
        try:
            return _parse_hmdb_table_of_peaks(txt)
        except Exception:
            # fall back to two-column if HMDB block is malformed
            pass
    return _parse_two_column(txt)

def broaden_peaks(df, width=0.02, ppm_min=-1.0, ppm_max=14.0, nbins=3000): # width = 0.02 originally
    """Convert discrete peaks to a smooth pseudo-spectrum."""
    x = np.linspace(ppm_min, ppm_max, nbins)
    y = np.zeros_like(x)
    for p, i in zip(df["ppm"].to_numpy(), df["intensity"].to_numpy()):
        y += i * np.exp(-0.5 * ((x - p) / width)**2)
    return pd.DataFrame({"ppm": x, "intensity": y})


def bin_spectrum(df: pd.DataFrame, bins: np.ndarray, cfg: dict) -> np.ndarray:
    nbins = len(bins) - 1
    vec = np.zeros(nbins, dtype=float)
    idx = np.digitize(df["ppm"].values, bins, right=True) - 1
    valid = (idx >= 0) & (idx < nbins)
    for i, inten in zip(idx[valid], df["intensity"].values[valid]):
        if math.isfinite(inten): vec[i] += float(inten)

    # normalization
    norm_mode = str(cfg.get("normalize","max")).lower()
    if norm_mode in ("max","peak"):
        m = vec.max()
        if m > 0: vec = vec / m
    elif norm_mode in ("l2","euclid"):
        s = np.linalg.norm(vec)
        if s > 0: vec = vec / s

    # log transform (after normalization is also fine for our use)
    if bool(cfg.get("log_transform", False)):
        vec = np.log1p(np.maximum(vec, 0))
    return vec

def prepare_input(vec: np.ndarray) -> np.ndarray:
    X = vec.copy()[None, :]   # (1, NBINS)
    X = X[..., None]          # (1, NBINS, 1)
    return X

# ====================== UI ======================

st.set_page_config(page_title="Calibrated H-NMR Classifier", layout="centered")
st.title("Calibrated H-NMR Classifier")
st.caption("Loads your baseline model and applies temperature scaling for trustworthy confidence estimates.")

with st.sidebar:
    st.header("Settings")
    tau = st.slider("Abstain threshold (τ)", 0.0, 1.0, 0.60, 0.01)

# persist file bytes
if "raw_bytes" not in st.session_state:
    st.session_state["raw_bytes"] = None

st.subheader("Upload spectrum (TXT/CSV/TSV)")
upload = st.file_uploader("Two columns: ppm, intensity (or HMDB TXT with 'Table of Peaks')", type=["txt","tsv","csv"], key="upload_1")
st.write("— or —")
local_path = st.text_input("Path on this machine (e.g., /Users/alfred/Desktop/example.txt)")

if upload is not None:
    st.session_state["raw_bytes"] = upload.getvalue()
elif local_path.strip():
    try:
        with open(local_path, "rb") as f:
            st.session_state["raw_bytes"] = f.read()
    except Exception as e:
        st.error(f"Failed to read file: {e}")

raw = st.session_state["raw_bytes"]

if raw is None:
    st.info("Upload a peaklist or enter a local path to continue.")
    st.stop()

# show sample and parse
st.caption(f"File size: {len(raw)} bytes")
st.code(raw[:400].decode("utf-8", errors="ignore"))

df = parse_bytes(raw)
df = broaden_peaks(df) 
if df.empty:
    st.error("No valid peaks parsed. Need a 'ppm' column and optional 'intensity' column, or an HMDB 'Table of Peaks'.")
    st.stop()

# preview
st.subheader("Preview")
st.dataframe(df.head(10))

fig1, ax1 = plt.subplots()
ax1.vlines(df["ppm"], [0], df["intensity"], linewidth=1)
ax1.invert_xaxis()
ax1.set_xlabel("ppm"); ax1.set_ylabel("intensity")
ax1.set_title("Uploaded peak list")
st.pyplot(fig1)

# ========== LOAD → BIN → PREDICT (auto-run with logs) ==========

st.subheader("Step 1: Load artifacts")
log_box = st.empty()
with st.spinner("Loading model & artifacts…"):
    try:
        predictor, bins, cfg, labels, load_logs = load_artifacts_with_logs()
        # print logs
        for line in load_logs:
            log_box.write(line)
        try:
            model_input_shape = predictor.model.input_shape
        except Exception:
            model_input_shape = "unknown"
        st.success(f"Artifacts loaded. model.input_shape={model_input_shape} · n_bins={len(bins)-1}")
    except Exception as e:
        st.error("Failed to load model artifacts.")
        st.exception(e)
        st.stop()

st.subheader("Step 2: Bin & prepare input")
try:
    vec = bin_spectrum(df, bins, cfg)
    X = prepare_input(vec)
    st.write(f"vector length = {len(vec)} | nonzero bins = {(vec>0).sum()} | X shape = {X.shape}")
    try:
        expected_bins = predictor.model.input_shape[1]
        st.write(f"expected_bins from model = {expected_bins}")
        if expected_bins is not None and expected_bins != len(vec):
            st.error(f"Input length ({len(vec)}) != expected ({expected_bins}). "
                     "Bins/config mismatch with model.")
            st.stop()
    except Exception:
        pass
except Exception as e:
    st.error("Binning / input prep failed.")
    st.exception(e)
    st.stop()

st.subheader("Step 3: Predict")
try:
    with st.spinner("Running prediction…"):
        out = predictor.predict(X, tau=tau)
    probs   = out["probs"][0]
    top_idx = int(out["top"][0])
    conf    = float(out["conf"][0])
    abstain = bool(out["abstain"][0])
    st.success("Prediction finished.")
except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.stop()

# label mapping
if labels and isinstance(labels, (list, tuple)) and top_idx < len(labels):
    top_label = labels[top_idx]
else:
    top_label = f"class_{top_idx}"

st.subheader("Result")
if abstain:
    st.warning(f"Uncertain prediction (max p = {conf:.2f} < τ={tau:.2f}). Please review.")
else:
    st.success(f"Predicted: **{top_label}** · Confidence: **{conf:.2f}**")

# probabilities
fig2, ax2 = plt.subplots()
xs = np.arange(len(probs))
ax2.bar(xs, probs)
ax2.set_xlabel("Class index")
ax2.set_ylabel("Calibrated probability")
ax2.set_title("Class probabilities (calibrated)")
if labels and isinstance(labels, (list, tuple)):
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(l) for l in labels], rotation=45, ha="right")
st.pyplot(fig2)

# binned spectrum
fig3, ax3 = plt.subplots()
centers = 0.5 * (bins[:-1] + bins[1:])
ax3.plot(centers, vec, linewidth=1)
ax3.invert_xaxis()
ax3.set_xlabel("ppm (binned)")
ax3.set_ylabel("normalized intensity")
ax3.set_title("Binned spectrum used for inference")
st.pyplot(fig3)
