# Automated Metabolite Family Identification from Proton NMR Data

## How to run

```
streamlit run streamlit_app.py
```

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
└── demo.mp4 - Screen recorded demo of how the webapp works

```


