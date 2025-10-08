#set text(10pt, font: "Arial")

#set page(header: text(8pt)[#align(right)[DDLS 2025 \ Alfred Larsson]])

#let quote2(body) = box(fill: gray.lighten(80%), inset: 8pt, radius: 4pt)[
    #set text(style: "italic")
    #body
]

#show link: set text(fill: blue)
#text(18pt)[*Project plan*]

= Background

Nuclear Magnetic Resonance (NMR) spectroscopy is an analytical technique that exploits the magnetic properties of atomic nuclei to study molecular structure. When placed in a strong magnetic field, nuclei with spin (such as #super[1]H, #super[13]C) can absorb radiofrequency energy and resonate at characteristic frequencies. ¹H-NMR (proton NMR) is the most common type, since nearly all organic molecules contain hydrogen. Each unique chemical environment around a hydrogen atom causes a slightly different resonance frequency (chemical shift), producing peaks in the spectrum. Peak splitting (multiplets) reveals information about neighboring atoms, while intensities reflect the number of protons, enabling detailed structural interpretation.

= Scientific question
Can we use supervised machine learning to classify small molecules into broad metabolite families (e.g., amino acids, sugars, lipids, nucleotides) based on their ¹H-NMR peaklist spectra?

= Scientific relevance
Metabolomics relies on NMR for rapid, non-destructive profiling of metabolites in biological samples. Assigning metabolites to broad chemical families from spectral data is useful for:

- Early biomarker discovery in precision medicine.

- Automated metabolite identification pipelines.

- Reducing manual spectral interpretation workload.

= Dataset

*Main dataset*

Source: Human Metabolome Database (HMDB, #link("https://hmdb.ca/downloads")[hmdb.ca/downloads])

License and terms of use:

#quote2[HMDB is offered to the public as a freely available resource. Use and re-distribution of the data, in whole or in part, for commercial purposes requires explicit permission of the authors and explicit acknowledgment of the source material (HMDB) and the original publication (see below). We ask that users who download significant portions of the database cite the HMDB paper in any resulting publications.]

*Files:*

- Raw NMR Spectra Peaklist Files (TXT) - ppm/intensity peaklists for many metabolites.

- All Metabolites Metadata (XML) - compound metadata, structures (SMILES, InChI), and classifications (superclass, class, subclass).

*Linking:* TXT filenames contain HMDB IDs (e.g., HMDB0000161) that map directly to entries in the XML metadata.

*Subset:* 500-1000 metabolites covering at least 4-5 broad families (Amino acids, Sugars, Lipids, Nucleotides, Organic acids).

*Supplemental datasets (if needed)*

Biological Magnetic Resonance Bank (BMRB)


= Exploration Goals

Before modeling, we will explore:

- Spectral structure: How many peaks per compound? Distribution of ppm ranges across classes?

- Class balance: Are families equally represented, or will we need class weighting / balancing?

- Spectral similarity: Do spectra from the same class cluster in PCA space?

- Technical variation: Are there solvent/field strength differences that need normalization?


= Proposed Model & Evaluation

*Baseline Model*

- Random Forest classifier on binned spectra (e.g., 0.02 ppm buckets).

- Logistic Regression as a second simple baseline.

*Improved Models*

- Support Vector Machine with RBF kernel.

- 1D Convolutional Neural Network (CNN) treating spectra as a sequence.

- Autoencoder for feature reduction, then classification.

*Evaluation*

- Metrics: Accuracy, weighted F1-score.

- Good performance definition: >70% accuracy across classes, better than random guessing and naive baselines.

- Cross-validation: 5-fold CV to ensure robustness.

= Accessibility

Wrapped pipeline in web application, which lets users upload NMR spectra to determine the metabolite group.

= Feasibility

*Expected runtime:*

- Preprocessing (binning, metadata join): \<5 minutes.

- Model training (Random Forest, SVM): seconds–minutes on CPU.

- CNN training: \<30 minutes on Colab GPU, still feasible on CPU with reduced dataset.

*Risks:*

- Large XML metadata file (≈6-9 GB) → handled by streaming parser, only extracting needed IDs.

- Class imbalance (e.g., lipids more abundant than amino acids) → mitigated by balanced sampling or weighting.

- Spectral variability (instrument/solvent effects) → normalized by scaling and binning.

*Overall:* Highly feasible in ~2-3 weeks.

#pagebreak()
= ChatGPT brainstorm

#link("https://chatgpt.com/share/68dd29b3-eae0-8007-89d9-0a72b6003ea6")[Link to ChatGPT brainstorming session.]

= Appendix
ChatGPT "Deep Research" report