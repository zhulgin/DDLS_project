# Trying to check if model can identify testosterone, using both HMDB and SDBS testosterone

#!/usr/bin/env python3
"""
classify_spectrum.py — Classify a new 1H-NMR peak list with your saved model.

Required (from training):
  - bins.npy
  - config.json
  - model.keras
  - optionally: label_encoder.pkl OR class_order.json  (for correct label order)

Input:
  - Two columns: ppm, intensity. CSV with header 'ppm,intensity' OR
    CSV/TXT without header (whitespace or comma separated).

Examples:
  python classify_spectrum.py peaks.txt
  python classify_spectrum.py peaks.txt --try-sum-norm --chem-aware
  python classify_spectrum.py peaks.txt --adaptive-aromatic-scrub
  python classify_spectrum.py peaks.txt --ensemble
  python classify_spectrum.py peaks.txt --reject-pmax 0.40 --reject-entr 0.95

Outputs:
  - Predicted class (or REJECT) + class probabilities
  - Debug: applied ppm shift, region fractions, chosen normalization, variant used
"""

import json
import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import joblib

# -------------------- Defaults & thresholds --------------------

FALLBACK_CLASS_NAMES = [
    "Nitrogenous & organic acids",   # 0
    "Lipids",                        # 1
    "Aromatics",                     # 2
    "Heterocycles & nucleotides",    # 3
    "Organic oxygen compounds",      # 4
]

# Strict defaults for blind use; override via CLI for sanity checks
DEFAULT_REJECT_PMAX = 0.50
DEFAULT_REJECT_ENTR = 0.88

EPS = 1e-12

# Diagnostic windows (ppm)
ALIPH_WINDOW = (0.0, 3.5)
OLEFIN_WINDOW = (4.5, 6.5)
AROM_WINDOW   = (6.5, 9.0)

# Solvent residuals (ppm ± delta)
SOLVENT_LINES = [
    (7.26, 0.05),  # CHCl3 (CDCl3)
    (2.50, 0.05),  # DMSO-d6
    (3.31, 0.06),  # MeOH-d4
    (4.87, 0.06),  # H2O/MeOH mix (varies)
    (2.05, 0.06),  # MeCN/acetone region
    (1.56, 0.08),  # H2O in D2O (varies)
    (0.00, 0.04),  # TMS
]

# Auto-referencing search params
DEFAULT_SHIFT_WINDOW = 0.50
DEFAULT_SHIFT_STEP   = 0.01

# Smoothing (in BIN units, not ppm)
DEFAULT_SMOOTH_SIGMA_BINS = 1.5


# -------------------- Artifacts & labels --------------------

def load_artifacts():
    if not os.path.exists("bins.npy"):  sys.exit("ERROR: Missing bins.npy")
    if not os.path.exists("config.json"):  sys.exit("ERROR: Missing config.json")
    if not os.path.exists("model.keras"):  sys.exit("ERROR: Missing model.keras")
    bins = np.load("bins.npy")
    cfg  = json.load(open("config.json"))
    model = tf.keras.models.load_model("model.keras")
    class_names = load_label_names()
    return bins, cfg, model, class_names

def load_label_names():
    if os.path.exists("label_encoder.pkl"):
        try:
            le = joblib.load("label_encoder.pkl")
            return list(le.classes_)
        except Exception:
            pass
    if os.path.exists("class_order.json"):
        try:
            return json.load(open("class_order.json"))
        except Exception:
            pass
    return FALLBACK_CLASS_NAMES


# -------------------- I/O & utilities --------------------

def load_peaks_flexible(path):
    # Try CSV with header
    try:
        arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
        if arr.ndim == 1 and arr.size == 2: arr = arr.reshape(1, 2)
        if arr.shape[1] >= 2:
            return arr[:,0].astype(np.float32), arr[:,1].astype(np.float32)
    except Exception:
        pass
    # Try CSV without header
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=float)
        if arr.ndim == 1 and arr.size == 2: arr = arr.reshape(1, 2)
        if arr.shape[1] >= 2:
            return arr[:,0].astype(np.float32), arr[:,1].astype(np.float32)
    except Exception:
        pass
    # Try whitespace-separated
    try:
        arr = np.loadtxt(path, dtype=float)
        if arr.ndim == 1 and arr.size == 2: arr = arr.reshape(1, 2)
        if arr.shape[1] >= 2:
            return arr[:,0].astype(np.float32), arr[:,1].astype(np.float32)
    except Exception:
        pass
    sys.exit(f"ERROR: Could not parse '{path}'. Expect two numeric columns: ppm, intensity.")

def mask_solvents(ppm, intensity):
    mask = np.ones_like(intensity, dtype=bool)
    for center, delta in SOLVENT_LINES:
        mask &= ~((ppm >= center - delta) & (ppm <= center + delta))
    return ppm[mask], intensity[mask]

def area_fraction_in_window(ppm, intensity, lo, hi):
    sel = (ppm >= lo) & (ppm <= hi)
    num = float(np.sum(intensity[sel]))
    den = float(np.sum(intensity) + EPS)
    return num / den

def rebin_peaks(ppm, intensity, bins):
    if ppm[0] > ppm[-1]:
        idx = np.argsort(ppm)
        ppm, intensity = ppm[idx], intensity[idx]
    binned, _ = np.histogram(ppm, bins=bins, weights=intensity)
    return binned.astype(np.float32)

def normalize_vector(x, mode="max"):
    if mode == "max":
        m = float(np.max(x));  return x / m if m > 0 else x
    elif mode == "sum":
        s = float(np.sum(x));  return x / s if s > 0 else x
    elif mode == "l2":
        n = float(np.linalg.norm(x));  return x / n if n > 0 else x
    return x

def gaussian_kernel_1d(sigma_bins, radius_factor=3):
    if sigma_bins is None or sigma_bins <= 0: return None
    r = max(1, int(radius_factor * sigma_bins))
    x = np.arange(-r, r+1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma_bins))**2)
    k /= np.sum(k)
    return k

def smooth_bins(x, sigma_bins=1.5):
    k = gaussian_kernel_1d(sigma_bins)
    if k is None: return x
    pad = len(k)//2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, k, mode="valid").astype(np.float32)

def normalized_entropy(probs):
    k = len(probs)
    ent = -np.sum(probs * np.log(np.clip(probs, EPS, 1.0)))
    return float(ent / np.log(k))


# -------------------- Adaptive aromatic scrub --------------------

def adaptive_aromatic_scrub(ppm, intensity, aliph_frac, arom_frac,
                            lo=6.8, hi=7.6, aliph_thr=0.75, arom_thr=0.12):
    """
    If spectrum is strongly aliphatic with very low aromatic fraction,
    zero out the 6.8–7.6 ppm band (often solvent tail region) to avoid
    spurious 'Aromatics' bias. Returns possibly-modified arrays.
    """
    if aliph_frac > aliph_thr and arom_frac < arom_thr:
        mask = ~((ppm >= lo) & (ppm <= hi))
        return ppm[mask], intensity[mask]
    return ppm, intensity


# -------------------- Auto-referencing (shift search) --------------------

def best_shift(ppm, intensity, bins, model, log_transform=False,
               normalize_mode="max", shift_window=DEFAULT_SHIFT_WINDOW,
               shift_step=DEFAULT_SHIFT_STEP, smooth_sigma_bins=DEFAULT_SMOOTH_SIGMA_BINS):
    shifts = np.arange(-shift_window, shift_window + 1e-9, shift_step)
    best_conf = -1.0
    best_shift_val = 0.0
    input_is_cnn = (len(model.input_shape) == 3)

    for s in shifts:
        x = rebin_peaks(ppm + s, intensity, bins)
        x = smooth_bins(x, sigma_bins=smooth_sigma_bins)
        if log_transform:
            x = np.log1p(np.maximum(x, 0.0))
        x = normalize_vector(x, normalize_mode)
        x_in = x.reshape(1, -1, 1) if input_is_cnn else x.reshape(1, -1)
        probs = model.predict(x_in, verbose=0).ravel()
        conf = float(np.max(probs))
        if conf > best_conf:
            best_conf = conf
            best_shift_val = s

    return best_shift_val, best_conf


# -------------------- Chem-aware reweight (optional) --------------------

def chem_aware_reweight(probs, class_names, aliph_frac, olefin_frac, arom_frac):
    """
    If very little aromatic content, gently down-weight 'Aromatics' class
    and renormalize. Conservative; only acts when arom_frac is tiny.
    """
    if class_names is None: return probs
    try:
        arom_idx = class_names.index("Aromatics")
    except ValueError:
        return probs
    if arom_frac < 0.10 and olefin_frac < 0.20:
        p = probs.copy()
        p[arom_idx] *= 0.6
        s = float(np.sum(p))
        if s > 0: p /= s
        return p
    return probs


# -------------------- Prediction helpers --------------------

def predict_with_norm(model, x_vec, norm_mode, is_cnn):
    xx = normalize_vector(x_vec, norm_mode)
    X = xx.reshape(1, -1, 1) if is_cnn else xx.reshape(1, -1)
    p = model.predict(X, verbose=0).ravel()
    return p, float(np.max(p)), int(np.argmax(p))


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="Classify 1H-NMR peak list with saved model.")
    ap.add_argument("peaks_path", help="Path to peaks file (two columns: ppm,intensity).")
    ap.add_argument("--print-labels", action="store_true", help="Print label order and exit.")
    ap.add_argument("--no-solvent-mask", action="store_true", help="Disable solvent residual masking.")
    ap.add_argument("--shift-window", type=float, default=DEFAULT_SHIFT_WINDOW, help="Auto-referencing window in ppm.")
    ap.add_argument("--shift-step", type=float, default=DEFAULT_SHIFT_STEP, help="Step size for shift search in ppm.")
    ap.add_argument("--smooth", type=float, default=DEFAULT_SMOOTH_SIGMA_BINS, help="Gaussian smoothing sigma in BINS.")
    ap.add_argument("--try-sum-norm", action="store_true", help="Also try sum-normalization, pick higher-confidence.")
    ap.add_argument("--chem-aware", action="store_true", help="Chem-aware reweight (down-weight Aromatics if aromatic fraction is tiny).")
    ap.add_argument("--adaptive-aromatic-scrub", action="store_true",
                    help="If aliphatic is dominant & aromatic tiny, remove 6.8–7.6 ppm band before binning.")
    ap.add_argument("--ensemble", action="store_true",
                    help="Evaluate (scrub off/on) × (norm cfg/sum) and pick the most confident result.")
    ap.add_argument("--reject-pmax", type=float, default=DEFAULT_REJECT_PMAX, help="Reject if max prob < this.")
    ap.add_argument("--reject-entr", type=float, default=DEFAULT_REJECT_ENTR, help="Reject if norm. entropy > this.")
    args = ap.parse_args()

    bins, cfg, model, class_names = load_artifacts()
    if args.print_labels:
        print("Label order used by the model:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
        sys.exit(0)

    input_is_cnn = (len(model.input_shape) == 3)

    # Load spectrum
    ppm, intensity = load_peaks_flexible(args.peaks_path)

    # Region diagnostics BEFORE masking/shifting
    aliph_frac_raw = area_fraction_in_window(ppm, intensity, *ALIPH_WINDOW)
    olefin_frac_raw = area_fraction_in_window(ppm, intensity, *OLEFIN_WINDOW)
    arom_frac_raw   = area_fraction_in_window(ppm, intensity, *AROM_WINDOW)

    # Solvent masking
    if args.no_solvent_mask:
        ppm2, inten2 = ppm, intensity
    else:
        ppm2, inten2 = mask_solvents(ppm, intensity)

    # Optional adaptive aromatic scrub (pre-binning)
    if args.adaptive_aromatic_scrub:
        ppm3, inten3 = adaptive_aromatic_scrub(ppm2, inten2, aliph_frac_raw, arom_frac_raw)
        scrub_used = True
    else:
        ppm3, inten3 = ppm2, inten2
        scrub_used = False

    # Auto-reference (shift search)
    best_s, _ = best_shift(
        ppm3, inten3, bins, model,
        log_transform=cfg.get("log_transform", False),
        normalize_mode=cfg.get("normalize", "max"),
        shift_window=args.shift_window, shift_step=args.shift_step,
        smooth_sigma_bins=args.smooth
    )

    # Rebin once at best shift, then smooth & optional log
    def make_x(ppm_arr, inten_arr):
        x = rebin_peaks(ppm_arr + best_s, inten_arr, bins)
        x = smooth_bins(x, sigma_bins=args.smooth)
        if cfg.get("log_transform", False):
            x = np.log1p(np.maximum(x, 0.0))
        return x

    x_base = make_x(ppm3, inten3)

    # Prediction variants
    norm_cfg = cfg.get("normalize", "max")
    variants = []

    def add_variant(name, xvec, norm_mode, do_scrub):
        probs, pmax, pred_idx = predict_with_norm(model, xvec, norm_mode, input_is_cnn)
        variants.append({
            "name": name,
            "probs": probs,
            "pmax": pmax,
            "pred_idx": pred_idx,
            "norm": norm_mode,
            "scrub": do_scrub
        })

    if args.ensemble:
        # (scrub off/on) × (norm cfg/sum)
        # scrub OFF path
        x_no_scrub = make_x(ppm2, inten2)
        add_variant("no_scrub+norm_cfg", x_no_scrub, norm_cfg, False)
        if args.try_sum_norm:
            add_variant("no_scrub+sum",     x_no_scrub, "sum",  False)
        # scrub ON path
        ppm_scr, inten_scr = adaptive_aromatic_scrub(ppm2, inten2, aliph_frac_raw, arom_frac_raw)
        x_scrub = make_x(ppm_scr, inten_scr)
        add_variant("scrub+norm_cfg", x_scrub, norm_cfg, True)
        if args.try_sum_norm:
            add_variant("scrub+sum",     x_scrub, "sum",  True)
    else:
        add_variant("base+norm_cfg", x_base, norm_cfg, scrub_used)
        if args.try_sum_norm:
            add_variant("base+sum",     x_base, "sum",  scrub_used)

    # Pick the variant with highest confidence
    best = max(variants, key=lambda d: d["pmax"])
    probs = best["probs"].copy()
    p_max = best["pmax"]
    pred_idx = best["pred_idx"]
    chosen_norm = best["norm"]
    scrub_used_final = best["scrub"]

    # Optional chem-aware reweight (applied after variant selection)
    if args.chem_aware:
        probs = chem_aware_reweight(probs, class_names, aliph_frac_raw, olefin_frac_raw, arom_frac_raw)
        p_max = float(np.max(probs))
        pred_idx = int(np.argmax(probs))

    ent = normalized_entropy(probs)

    # Output
    print(f"Auto-referenced shift applied: {best_s:+.3f} ppm")
    print(f"Region fractions (raw): Aliphatic {aliph_frac_raw:.3f} | Olefinic {olefin_frac_raw:.3f} | Aromatic {arom_frac_raw:.3f}")
    print(f"Normalization chosen for final prediction: {chosen_norm}")
    print(f"Variant used: {'scrub' if scrub_used_final else 'no_scrub'} + {chosen_norm}")
    if args.chem_aware:
        print("Chem-aware reweighting: ENABLED")
    if args.no_solvent_mask:
        print("Solvent masking: DISABLED")
    if args.ensemble:
        # Print a compact variant summary
        vs = " | ".join([f"{v['name']} pmax={v['pmax']:.3f}" for v in variants])
        print(f"Ensemble variants: {vs}")
    print()

    # Decision
    reject_pmax = args.reject_pmax
    reject_entr = args.reject_entr
    if (p_max < reject_pmax) or (ent > reject_entr):
        print("Prediction: REJECT (none of the defined classes)")
        reasons = []
        if p_max < reject_pmax: reasons.append(f"low confidence (p_max={p_max:.3f})")
        if ent   > reject_entr: reasons.append(f"high uncertainty (entropy={ent:.2f})")
        if reasons:
            print("Reason → " + " or ".join(reasons))
    else:
        if pred_idx < len(class_names):
            pred_name = class_names[pred_idx]
            print(f"Prediction: {pred_name} (index {pred_idx})")
        else:
            print(f"Prediction: class index {pred_idx} (no name available)")

    # Probabilities
    print("\nClass probabilities:")
    for i, p in enumerate(probs):
        label = class_names[i] if i < len(class_names) else f"class_{i}"
        print(f"  {i}: {label:>28s}  {p:.3f}")

    print(f"\nMax prob: {p_max:.3f} | Normalized entropy: {ent:.2f}")


# -------------------- CLI --------------------

if __name__ == "__main__":
    main()
