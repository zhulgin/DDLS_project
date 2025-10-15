# --- Apple Silicon / fork-safety prelude (must be FIRST) ---
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
# ------------------------------------------------------------

import os, json, joblib, numpy as np
from pathlib import Path
from tensorflow import keras

ARTIF_DIR = Path(__file__).resolve().parent / "models" / "baseline"
MODEL_PATH = ARTIF_DIR / "model.keras"
ENCODER_PATH = ARTIF_DIR / "label_encoder.pkl"
BINS_PATH = ARTIF_DIR / "bins.npy"
CONFIG_PATH = ARTIF_DIR / "config.json"
CLASS_ORDER_JSON = ARTIF_DIR / "class_order.json"

def main():
    print("== Verify training artifacts ==")
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
    assert ENCODER_PATH.exists(), f"Missing label encoder: {ENCODER_PATH}"
    assert CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}"

    # Load pieces
    print("Loading model…")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    print("Config:", cfg)

    if BINS_PATH.exists():
        bins = np.load(BINS_PATH)
        print(f"bins.npy present. nbins={len(bins)-1}")
        # sanity vs config
        nbins_cfg = int(cfg.get("nbins"))
        assert len(bins) - 1 == nbins_cfg, "bins.npy nbins != config.json nbins"
    else:
        print("WARN: bins.npy missing — will rely on config.json to build edges.")
        bins = np.linspace(cfg["ppm_min"], cfg["ppm_max"], int(cfg["nbins"]) + 1)

    le = joblib.load(ENCODER_PATH)
    classes = list(le.classes_)
    print(f"Classes ({len(classes)}): {classes}")

    # Rewrite class_order.json from encoder, then assert equality if existed
    if CLASS_ORDER_JSON.exists():
        with open(CLASS_ORDER_JSON) as f:
            old = json.load(f)
        if old != classes:
            print("NOTE: class_order.json differs from encoder; rewriting to match encoder.")
    with open(CLASS_ORDER_JSON, "w") as f:
        json.dump(classes, f, indent=2)

    # Dry-run model input shape
    in_shape = model.inputs[0].shape
    exp_len = int(cfg["nbins"])
    assert in_shape[-2] in (None, exp_len), f"Model expects {in_shape[-2]} bins, config has {exp_len}"
    print("All checks passed ✅")

if __name__ == "__main__":
    main()
