# Automated Metabolite Family Identification from Proton NMR Data

## Overview

This project develops a machine learning pipeline to automatically classify metabolite compounds into five chemical groups based on ¹H-NMR spectroscopy data from the Human Metabolome Database. After evaluating convolutional neural networks (which failed with <25% accuracy) and ensemble methods, a tuned Random Forest classifier achieved 66.4% test accuracy on 847 compounds. Feature importance and SHAP analysis confirmed the model learned chemically meaningful patterns, with aliphatic regions (1.2-1.6 ppm) distinguishing lipids, aromatic signals (6-8 ppm) identifying aromatics, and alpha-to-heteroatom resonances (3.6 ppm) characterizing nitrogen/oxygen-containing compounds. A Streamlit web application enables real-time classification of uploaded NMR peak lists, providing predictions with confidence scores and class probability distributions.

## How to run

1. Install dependencies (requirements.txt)

2. Run
```
streamlit run streamlit_app.py
```

3. Choose H-NMR CSV/TXT peaklist file to classify

## Peaklist format

Peaklists are CSV/TXT files containing two columns: chemical shift (ppm) and intensity. 

Example: 
```
ppm,intensity
1.42,0.0912 	
1.44,0.2511 	
1.45,0.3889
```

Alternatively, peaklist files can be exported directly from Mestrenova by going to Save As > Script: NMR 1D Peak List (*.csv *.txt). In the Custom 1D CSV Export window, make sure Format says `{ppm},{intensity}`.

## Dependencies

List of dependencies can be found in `requirements.txt`

## File structure

```
├── code/
│    ├── CNN_attempt_figures/ - Figures from CNN model
│    ├── models/
│       ├── label_encoder.pkl
│       ├── preprocessing_params.json
│       ├── rf_tuned_model.pkl
│    ├── original_data/
│       ├── hmdb_nmr_peaklists - NMR peaklist data
│    ├── processed_data/
│       ├── feature_importance_data.csv
│       ├── group_counts.csv
│       ├── hmdb_subset_classes.csv
│       ├── hmdb_subset_super_groups.csv
│       ├── keep_ids_oned_h1.json
│       ├── nmr_features_with_groups.csv
│       ├── oned_h1_file_map.csv
│       ├── super_class_counts.csv
│    ├── RF_classifier_figures/
│    ├── shap_output/ - Figures from SHAP analysis
│    ├── test_spectra/
│       ├── 2-(5-benzyloxy-3-indolyl)ethylamine_HCl.txt
│       ├── 2-octenedioic_acid_hmdb.txt
│       ├── CA8353_crude_new.csv
│       ├── CA8353_crude_new.mnova
│       ├── CA8396.csv
│       ├── CA8396.mnova
│       ├── dmt.txt
│       ├── nandrolone.txt
│       ├── testosterone_hmdb.txt
│       ├── testosterone_sdbs.txt
│    ├── classify_spectrum.py
│    ├── CNN_attempt.ipynb - Trying CNN models
│    ├── data_extraction.ipynb - Extracting data from HMDB files
│    ├── RF_classifier.ipynb - Building RF model
│    ├── shap_analysis.ipynb - SHAP analysis of RF model
│    ├── streamlit_app.py - Streamlit app
│    └── streamlit_app_no-mnova.py - Streamlit app (old)
├── Final report/ - _Typst files for final report_
│    ├── fig/ - _Figures for final report_
│    ├── main.typ - Main document
│    ├── abstract.typ
│    ├── appendix_figures.typ
│    ├── background.typ
│    ├── conclusion.typ
│    ├── dataset_summary.typ
│    ├── method.typ
│    ├── results.typ
│    ├── references.bib - List of references
├── Project plan/ - _Typst files for project plan_
│    ├──deep-research-report.pdf
│    ├──main.typ
│    ├──main_v2.pdf
│    ├──main_v2.typ
│    ├──project-plan-alfred-larsson_v1.pdf
│    ├──project-plan-alfred-larsson_v2_deep-research.pdf
├── README.md
├── requirements.txt
├── LICENSE
└── demo.mp4 - Screen recorded demo of how the webapp works

```


