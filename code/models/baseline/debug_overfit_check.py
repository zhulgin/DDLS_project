# --- Apple Silicon / fork-safety prelude (must be FIRST) ---
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
# ------------------------------------------------------------


# debug_overfit_check.py
import os, io, json, joblib, numpy as np, tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# --- config & artifacts ---
cfg = json.load(open("config.json"))
bins = np.load("bins.npy")
n_bins = bins.size - 1
print(f"[ARTIFACTS] nbins from file: {n_bins} | cfg: {cfg}")

label_encoder = joblib.load("label_encoder.pkl")
classes = getattr(label_encoder, "classes_", None)
print(f"[LABELS] classes ({len(classes)}): {list(classes)}")

scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None
print(f"[SCALER] present: {scaler is not None}")

model = tf.keras.models.load_model("model.keras")
model_in = tuple(getattr(model, "input_shape", None) or ())
print(f"[MODEL] input_shape: {model_in}, outputs: {model.output_shape}")

# --- helpers ---
import io, csv, pandas as pd, numpy as np
from pathlib import Path

def preprocess_from_txt(path):
    txt = Path(path).read_text(errors="ignore")

    # --- detect delimiter (comma/semicolon/tab/space) ---
    sample = txt[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        sep = dialect.delimiter
    except Exception:
        sep = None  # fall back to whitespace

    # --- read with/without header robustly ---
    def _read(header):
        return pd.read_csv(
            io.StringIO(txt),
            sep=sep,
            comment="#",
            header=header,       # 0 if header row, None if not
            engine="python",
        )

    try:
        df = _read(0)  # try with header ("ppm,intensity")
        # If header was wrong (strings ended up in data), coerce numerics:
        tmp = df.apply(pd.to_numeric, errors="coerce")
        if tmp.isna().all().all():  # nothing parsed -> try no header
            df = _read(None)
    except Exception:
        df = _read(None)

    # keep only first two columns and coerce numerics
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("No numeric ppm/intensity data found after parsing.")

    a = df.to_numpy(dtype=np.float32)
    col0, col1 = a[:, 0], a[:, 1]

    # --- figure out which column is ppm (typical ppm range ~ -5..20) ---
    def looks_like_ppm(v):
        vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
        return (vmin > -10) and (vmax < 25)

    if looks_like_ppm(col0) and not looks_like_ppm(col1):
        ppm, inten = col0, col1
    elif looks_like_ppm(col1) and not looks_like_ppm(col0):
        ppm, inten = col1, col0
    else:
        # fallback: assume first is ppm
        ppm, inten = col0, col1

    # keep only ppm within training bins
    ppm_min, ppm_max = bins[0], bins[-1]
    m = (ppm >= ppm_min) & (ppm <= ppm_max)
    ppm, inten = ppm[m], inten[m]

    # bin
    hist, _ = np.histogram(ppm, bins=bins, weights=inten)
    x = hist.astype(np.float32)

    # optional log + normalization from cfg
    if cfg.get("log_transform", False):
        x = np.log1p(np.maximum(x, 0))
    norm = cfg.get("normalize")
    if norm == "max":
        m = float(x.max())
        x = x / (m if m > 0 else 1.0)
    elif norm == "z" and scaler is not None:
        x = scaler.transform(x[None, :])[0].astype(np.float32)

    # optional smoothing if extremely sparse
    if (x > 0).sum() < 50:
        try:
            from scipy.ndimage import gaussian_filter1d
            x = gaussian_filter1d(x, sigma=1.0)
        except Exception:
            pass

    return x.reshape(1, -1, 1)


def raw_predict(x1):
    # pre-calibration logits/probs
    probs = model.predict(x1, verbose=0)[0]
    return probs

# ========== CHECK 1: bins and model agree ==========
expected_bins = model.layers[0].input_shape[1] if hasattr(model.layers[0], "input_shape") else n_bins
if expected_bins != n_bins:
    print(f"[FAIL] nbins mismatch: model expects {expected_bins}, bins.npy has {n_bins}")
else:
    print("[OK] nbins match model input.")

# ========== CHECK 2: run on a KNOWN training file ==========
# Point this to one spectrum you KNOW was in training (TXT path)
KNOWN_TRAIN_TXT = os.environ.get("KNOWN_TRAIN_TXT", "")
if KNOWN_TRAIN_TXT and os.path.exists(KNOWN_TRAIN_TXT):
    X = preprocess_from_txt(KNOWN_TRAIN_TXT)
    nz = int((X>0).sum())
    print(f"[INPUT] shape={X.shape} nonzero_bins={nz} dtype={X.dtype} max={float(X.max()):.4f}")
    probs = raw_predict(X)
    print(f"[RAW PROBS] max={float(probs.max()):.3f} argmax={int(np.argmax(probs))} probs={np.round(probs,3)}")

    # Try without/with scaler if co
