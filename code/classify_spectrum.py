#!/usr/bin/env python3
"""
H-NMR Spectrum Classification Script
=====================================
Classify H-NMR peak list files ("offline") using a trained Random Forest model.

Usage:
    python3 classify_spectrum.py path/to/spectrum.txt
    python3 classify_spectrum.py path/to/spectrum.txt --verbose
"""

import sys
import pickle
import json
import numpy as np
import re
from pathlib import Path
import argparse

# --- CONFIGURATION ---
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "rf_tuned_model.pkl"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
PARAMS_FILE = MODEL_DIR / "preprocessing_params.json"

# --- HELPER FUNCTIONS ---

def parse_nmr_peak_file(filepath, ppm_min, ppm_max):
    """
    Parse a CSV file with ppm,intensity format.
    Expected format:
        ppm,intensity
        7.20075,6582.310017
        7.181023,6006.043662
        ...
    """
    ppm_list, inten_list = [], []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Skip header row (contains 'ppm' or 'intensity')
            if i == 0 and ('ppm' in line.lower() or 'intensity' in line.lower()):
                continue
            
            # Split by comma
            parts = line.split(',')
            if len(parts) != 2:
                continue
            
            try:
                ppm = float(parts[0])
                inten = float(parts[1])
                
                # Filter by ppm range and positive intensity
                if ppm_min <= ppm <= ppm_max and inten > 0:
                    ppm_list.append(ppm)
                    inten_list.append(inten)
            except ValueError:
                # Skip lines that can't be parsed as floats
                continue
    
    return np.array(ppm_list, dtype=float), np.array(inten_list, dtype=float)

def peaks_to_vector(ppm, inten, bin_edges):
    """Convert peak list to binned vector."""
    n_bins = len(bin_edges) - 1
    vec = np.zeros(n_bins, dtype=float)
    
    if ppm.size == 0:
        return vec
    
    # Assign peaks to bins (max pooling)
    idx = np.digitize(ppm, bin_edges) - 1
    mask = (idx >= 0) & (idx < n_bins)
    
    for b, v in zip(idx[mask], inten[mask]):
        vec[b] = max(vec[b], v)
    
    # Apply log transform
    vec = np.log1p(vec)
    
    # Normalize
    max_val = vec.max()
    if max_val > 0:
        vec = vec / max_val
    
    return vec

def classify_spectrum(filepath, model, encoder, params, verbose=False):
    """
    Complete prediction pipeline for an H-NMR peak list file.
    
    Returns:
        dict with prediction results
    """
    if verbose:
        print(f"\nProcessing: {filepath}")
    
    # Parse the file
    ppm, inten = parse_nmr_peak_file(
        filepath, 
        params['ppm_min'], 
        params['ppm_max']
    )
    
    if ppm.size == 0:
        return {
            'error': 'No valid peaks found in file',
            'filename': str(filepath)
        }
    
    if verbose:
        print(f"Found {ppm.size} peaks in range {params['ppm_min']}-{params['ppm_max']} ppm")
    
    # Convert to binned vector
    bin_edges = np.array(params['bin_edges'])
    vec = peaks_to_vector(ppm, inten, bin_edges)
    
    # Normalize using training statistics
    mean = np.array(params['normalization_mean'])
    std = np.array(params['normalization_std'])
    vec_norm = (vec - mean) / std
    
    # Reshape for prediction
    X = vec_norm.reshape(1, -1)
    
    # Make prediction
    pred_encoded = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    
    # Decode prediction
    pred_class = encoder.inverse_transform([pred_encoded])[0]
    confidence = float(pred_proba[pred_encoded])
    
    # Get all class probabilities
    class_probabilities = {
        encoder.inverse_transform([i])[0]: float(prob) 
        for i, prob in enumerate(pred_proba)
    }
    
    return {
        'filename': str(filepath),
        'predicted_class': pred_class,
        'confidence': confidence,
        'all_probabilities': class_probabilities,
        'n_peaks': int(ppm.size)
    }

# --- MAIN SCRIPT ---

def main():
    parser = argparse.ArgumentParser(
        description='Classify H-NMR spectra into compound groups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 classify_spectrum.py spectrum.txt
  python3 classify_spectrum.py spectrum.txt --verbose
  python3 classify_spectrum.py data/HMDB0002068.txt -v
        """
    )
    parser.add_argument('spectrum_file', help='Path to H-NMR peak list file')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Show detailed processing information')
    
    args = parser.parse_args()
    
    # Check if input file exists
    spectrum_path = Path(args.spectrum_file)
    if not spectrum_path.exists():
        print(f"Error: File not found: {spectrum_path}")
        sys.exit(1)
    
    # Check if model files exist
    if not MODEL_FILE.exists():
        print(f"Error: Model file not found: {MODEL_FILE}")
        print("Please ensure you have run the training notebook and saved the model.")
        sys.exit(1)
    
    if not ENCODER_FILE.exists():
        print(f"Error: Label encoder not found: {ENCODER_FILE}")
        sys.exit(1)
    
    if not PARAMS_FILE.exists():
        print(f"Error: Preprocessing parameters not found: {PARAMS_FILE}")
        sys.exit(1)
    
    # Load model and parameters
    if args.verbose:
        print("Loading model...")
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    with open(ENCODER_FILE, 'rb') as f:
        encoder = pickle.load(f)
    
    with open(PARAMS_FILE, 'r') as f:
        params = json.load(f)
    
    if args.verbose:
        print("✓ Model loaded successfully")
    
    # Make prediction
    result = classify_spectrum(spectrum_path, model, encoder, params, verbose=args.verbose)
    
    # Display results
    if 'error' in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("CLASSIFICATION RESULT")
    print("="*60)
    print(f"File: {result['filename']}")
    print(f"Peaks found: {result['n_peaks']}")
    print(f"\nPredicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if args.verbose:
        print(f"\nAll class probabilities:")
        # Sort by probability descending
        sorted_probs = sorted(result['all_probabilities'].items(), 
                            key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_probs:
            bar = "█" * int(prob * 40)
            print(f"  {class_name:.<40} {prob:>6.1%} {bar}")
    
    print("="*60)

if __name__ == "__main__":
    main()