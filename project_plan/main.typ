#set text(font: "Arial")

#set page(header: text(8pt)[#align(right)[DDLS 2025 \ Alfred Larsson]])

#show link: set text(fill: blue)
#text(18pt)[*Project plan*]

= Background

Nuclear Magnetic Resonance (NMR) spectroscopy is an analytical technique that exploits the magnetic properties of atomic nuclei to study molecular structure. When placed in a strong magnetic field, nuclei with spin (such as #super[1]H, #super[13]C) can absorb radiofrequency energy and resonate at characteristic frequencies. ¹H-NMR (proton NMR) is the most common type, since nearly all organic molecules contain hydrogen. Each unique chemical environment around a hydrogen atom causes a slightly different resonance frequency (chemical shift), producing peaks in the spectrum. Peak splitting (multiplets) reveals information about neighboring atoms, while intensities reflect the number of protons, enabling detailed structural interpretation.

= Scientific Question
Can we use supervised machine learning to classify small molecules into broad metabolite families (e.g., amino acids, sugars, lipids, nucleotides) based on their ¹H-NMR peaklist spectra?

= Scientific relevance
Metabolomics relies on NMR for rapid, non-destructive profiling of metabolites in biological samples. Assigning metabolites to broad chemical families from spectral data is useful for:

- Early biomarker discovery in precision medicine.

- Automated metabolite identification pipelines.

- Reducing manual spectral interpretation workload.

= Dataset

Source: Human Metabolome Database (HMDB, hmdb.ca)

*Files:*

- Raw NMR Spectra Peaklist Files (TXT) - ppm/intensity peaklists for many metabolites.

- All Metabolites Metadata (XML) - compound metadata, structures (SMILES, InChI), and classifications (superclass, class, subclass).

*Linking:* TXT filenames contain HMDB IDs (e.g., HMDB0000161) that map directly to entries in the XML metadata.

*Subset:* 500-1000 metabolites covering at least 4-5 broad families (Amino acids, Sugars, Lipids, Nucleotides, Organic acids).

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



#pagebreak()
= Appendix

#link("https://chatgpt.com/share/68dd29b3-eae0-8007-89d9-0a72b6003ea6")[Link to ChatGPT brainstorming session.]