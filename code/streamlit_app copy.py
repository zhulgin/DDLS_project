"""
H-NMR Spectrum Classification Web App
======================================
Upload H-NMR peak lists and get compound class predictions.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import io

# --- CONFIGURATION ---
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "rf_tuned_model.pkl"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
PARAMS_FILE = MODEL_DIR / "preprocessing_params.json"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="H-NMR Classifier",
    page_icon="🧪",
    layout="centered"
)

# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_model():
    """Load model, encoder, and parameters (cached)."""
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_FILE, 'rb') as f:
        encoder = pickle.load(f)
    with open(PARAMS_FILE, 'r') as f:
        params = json.load(f)
    return model, encoder, params

def parse_csv_peaks(file_content, ppm_min, ppm_max):
    """
    Parse CSV file with ppm,intensity format.
    """
    ppm_list, inten_list = [], []
    
    # Decode if bytes
    if isinstance(file_content, bytes):
        file_content = file_content.decode('utf-8')
    
    lines = file_content.strip().split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Skip header
        if i == 0 and ('ppm' in line.lower() or 'intensity' in line.lower()):
            continue
        
        parts = line.split(',')
        if len(parts) != 2:
            continue
        
        try:
            ppm = float(parts[0])
            inten = float(parts[1])
            
            if ppm_min <= ppm <= ppm_max and inten > 0:
                ppm_list.append(ppm)
                inten_list.append(inten)
        except ValueError:
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
    
    # Log transform and normalize
    vec = np.log1p(vec)
    max_val = vec.max()
    if max_val > 0:
        vec = vec / max_val
    
    return vec

def classify_spectrum(file_content, model, encoder, params):
    """Complete prediction pipeline."""
    # Parse peaks
    ppm, inten = parse_csv_peaks(file_content, params['ppm_min'], params['ppm_max'])
    
    if ppm.size == 0:
        return None, "No valid peaks found in file"
    
    # Convert to binned vector
    bin_edges = np.array(params['bin_edges'])
    vec = peaks_to_vector(ppm, inten, bin_edges)
    
    # Normalize using training statistics
    mean = np.array(params['normalization_mean'])
    std = np.array(params['normalization_std'])
    vec_norm = (vec - mean) / std
    
    # Predict
    X = vec_norm.reshape(1, -1)
    pred_encoded = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    
    # Decode
    pred_class = encoder.inverse_transform([pred_encoded])[0]
    confidence = float(pred_proba[pred_encoded])
    
    # All probabilities
    class_probabilities = {
        encoder.inverse_transform([i])[0]: float(prob) 
        for i, prob in enumerate(pred_proba)
    }
    
    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'all_probabilities': class_probabilities,
        'n_peaks': int(ppm.size)
    }, None

# --- MAIN APP ---

# Title and description
st.title("🧪 H-NMR Spectrum Classifier")
st.markdown("""
Upload an H-NMR peak list (CSV format) to classify the compound into one of five groups:
- **Aromatics**
- **Heterocycles & nucleotides**
- **Lipids**
- **Nitrogenous & organic acids**
- **Organic oxygen compounds**
""")

# Load model
try:
    model, encoder, params = load_model()
    st.success("✓ Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.markdown("---")

# File upload
st.subheader("📁 Upload Spectrum")
uploaded_file = st.file_uploader(
    "Choose a CSV file with ppm,intensity format",
    type=['csv', 'txt'],
    help="Expected format: ppm,intensity (one peak per line)"
)

# Example format
with st.expander("ℹ️ Expected file format"):
    st.code("""ppm,intensity
7.20075,6582.310017
7.181023,6006.043662
7.161048,9861.65747
7.072613,4481.981926
...""", language="csv")

# Process uploaded file
if uploaded_file is not None:
    st.markdown("---")
    
    # Read file
    file_content = uploaded_file.read()
    
    # Classify
    with st.spinner("Analyzing spectrum..."):
        result, error = classify_spectrum(file_content, model, encoder, params)
    
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    # Display results
    st.subheader("📊 Classification Results")
    
    # Main prediction
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", result['predicted_class'])
    with col2:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    
    st.metric("Peaks Found", result['n_peaks'])
    
    # Probability distribution
    st.subheader("Class Probabilities")
    
    # Prepare data for plotting
    probs_df = pd.DataFrame(
        list(result['all_probabilities'].items()),
        columns=['Class', 'Probability']
    ).sort_values('Probability', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        probs_df,
        x='Probability',
        y='Class',
        orientation='h',
        color='Probability',
        color_continuous_scale='Blues',
        text=probs_df['Probability'].apply(lambda x: f'{x:.1%}')
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Probability",
        yaxis_title="",
        height=300,
        xaxis=dict(tickformat='.0%')
    )
    
    fig.update_traces(textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed probabilities table
    with st.expander("📋 Detailed probabilities"):
        detailed_df = pd.DataFrame(
            list(result['all_probabilities'].items()),
            columns=['Class', 'Probability']
        ).sort_values('Probability', ascending=False)
        detailed_df['Probability'] = detailed_df['Probability'].apply(lambda x: f"{x:.2%}")
        st.dataframe(detailed_df, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    H-NMR Classification Model | Trained on HMDB metabolite data
</div>
""", unsafe_allow_html=True)