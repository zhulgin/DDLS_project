#!/usr/bin/env python3
"""
classify_spectrum.py — Classify a new 1H-NMR peak list with your saved CNN
Inputs expected:
  - bins.npy        (bin edges saved after training)
  - config.json     (your preprocessing settings; normalize/log flags)
  - model.keras     (your trained model)
  - peaks CSV       (two columns with header: ppm,intensity), e.g. from mnova_to_peaks.py

Usage:
  python classify_spectrum.py path/to/peaks_for_model.csv

Outputs:
  - Prints predicted class (or REJECT) and class probabilities
"""

import json
import sys
import os
import numpy as np
import tensorflow as tf

# ======== EDIT THIS if your class index->name mapping differs ========
CLASS_NAMES = [
    "Nitrogenous & organic acids",   # index 0
    "Lipids",                        # index 1
    "Aromatics",                     # index 2
    "Heterocycles & nucleotides",    # index 3
    "Organic oxygen compounds",      # index 4
]
# ======== REJECTION THRESHOLDS (tune on validation set) ========
CONFIDENCE_THRESHOLD = 0.55   # if max(prob) < this -> reject
ENTROPY_THRESHOLD    = 0.75   # if normalized entropy > this -> reject
EPS = 1e-12

def load_artifacts():
    if not (os.path.exists("bins.npy") and os.path.exists("config.json") and os.path.exists("model.keras")):
        sys.exit("ERROR: Missing one or more required files: bins.npy, config.json, model.keras")
    bins   = np.load("bins.npy")
    cfg    = json.load(open("config.json"))
    model  = tf.keras.models.load_model("model.keras")
    return bins, cfg, model

def load_peaks_csv(path):
    """
    Expects a CSV with header 'ppm,intensity' and numeric values.
    (Produced by mnova_to_peaks.py)
    """
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
    except Exception as e:
        sys.exit(f"ERROR reading peaks CSV '{path}': {e}")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        sys.exit(f"ERROR: '{path}' must have two columns: ppm,intensity")
    return arr[:, :2]  # (ppm, intensity)

def bin_peaks(peaks, bins):
    """Convert (ppm, intensity) pairs into a fixed-length binned spectrum."""
    ppm = peaks[:, 0]
    inten = peaks[:, 1]
    # weighted histogram by intensity into the predefined bins
    x, _ = np.histogram(ppm, bins=bins, weights=inten)
    return x.astype(np.float32)

def preprocess_vector(x, cfg):
    """Apply the same normalization choices used during training."""
    if cfg.get("log_transform", False):
        x = np.log1p(np.maximum(x, 0.0))
    norm = cfg.get("normalize")
    if norm == "max":
        m = float(np.max(x)) if np.max(x) > 0 else 1.0
        x = x / m
    elif norm == "l2":
        denom = float(np.linalg.norm(x)) or 1.0
        x = x / denom
    return x

def normalized_entropy(probs):
    """Shannon entropy normalized to [0,1] (1 = maximally uncertain)."""
    k = len(probs)
    ent = -np.sum(probs * np.log(probs + EPS))
    return float(ent / np.log(k))

def main(peaks_csv):
    bins, cfg, model = load_artifacts()
    peaks = load_peaks_csv(peaks_csv)

    # Bin & preprocess
    x = bin_peaks(peaks, bins)
    x = preprocess_vector(x, cfg)
    X = x.reshape(1, -1, 1)  # (batch, length, channels) for 1D-CNN

    # Predict
    probs = model.predict(X, verbose=0)[0]
    p_max = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    ent = normalized_entropy(probs)

    # Reject / accept decision
    if (p_max < CONFIDENCE_THRESHOLD) or (ent > ENTROPY_THRESHOLD):
        print("\nPrediction: REJECT (none of the defined classes)")
        print(f"Reason → low confidence (p_max={p_max:.3f}) or high uncertainty (entropy={ent:.2f})")
    else:
        # safety if CLASS_NAMES mismatched length
        if pred_idx < len(CLASS_NAMES):
            pred_name = CLASS_NAMES[pred_idx]
            print(f"\nPrediction: {pred_name} (index {pred_idx})")
        else:
            print(f"\nPrediction: class index {pred_idx} (no name available)")

    # Always show probabilities
    # If CLASS_NAMES length matches, print labeled probs for convenience
    print("\nClass probabilities:")
    for i, p in enumerate(probs):
        label = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"  {i}: {label:>28s}  {p:.3f}")

    print(f"\nMax prob: {p_max:.3f} | Normalized entropy: {ent:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_spectrum.py peaks_for_model.csv")
        sys.exit(1)
    main(sys.argv[1])
