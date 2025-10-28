#let title = "Automated Metabolite Family Identification from Proton NMR Data"

#set text(11pt, font: "New Computer Modern")
#set document(title: title)
#set page(
  header: text[
    #set align(right)
    #set text(9pt)
    _#[#title]_\
    DDLS 2025\
    Alfred Larsson
  ]
)

#show "HNMR": [#super[1]H NMR]
#let rawbox(body) = box(fill: gray.lighten(85%), inset: 8pt, radius: 6pt)[#body]
#show figure.where(kind: "code"): set figure(supplement: "Code")
#show figure.caption: set text(10pt, style: "italic")

= Abstract
// (≤100 words): problem, method, results.

= Background
// Background and Motivation: why you chose this dataset/task.

Nuclear Magnetic Resonance (NMR) spectroscopy is an analytical technique that exploits the
magnetic properties of atomic nuclei to study molecular structure. When placed in a strong magnetic
field, nuclei with spin can absorb radiofrequency energy and resonate at
characteristic frequencies. HNMR is the most common type, since nearly all organic
molecules contain hydrogen, but NMR can be used for any atomic nuclei with spin, such as #super[13]C, #super[19]F and #super[15]N. In HNMR, each unique chemical environment around a hydrogen atom causes a slightly different resonance frequency (chemical shift), producing peaks in the spectrum. Peak splitting
(multiplets) reveals information about neighboring atoms, while intensities reflect the number of
protons, enabling detailed structural interpretation. HNMR is used routinely for exact structure determination of small molecules in organic chemistry due to its speed and ease of use (typically #sym.tilde.basic\5 min for a HNMR measurement) but also has applications in metabolomics, protein and nucleic acid studies, material science and medical imaging (MRI).



= Dataset Summary
// data source, preprocessing, splits, distributions.

HNMR data was retrieved from the Human Metabolome Database (HMDB) @hmdb. The HMDB data is made up of two parts: 

- A #raw(".xml") file containing various information for each compound in the database (HMDB ID, name and synonyms, InChI, description, molecular weight, CAS number, etc.) 

- A folder containing #raw(".txt") files (see @peaklist) with HNMR peak lists (and in some cases also multiplets and assignments) for a subset of the compounds. (See @peaklist)

#figure(
  caption: [Structure of a typical HNMR peaklist file. Three columns including peak number, chemical shift in ppm and peak height.],
  kind: "code",
  rawbox("Table of Peaks
No.   (ppm)	  Height
1      2.49     0.0970
2      2.50     0.1098
3      2.50     0.1134
...")
) <peaklist>

After filtering for compounds with available 1D HNMR spectra, the dataset included approximately 800 compounds with corresponding peak lists.

== Class definitions

The original HMDB hierarchical classification system was consolidated to create five primary compound groups suitable for machine learning classification:

+ Aromatics (127 samples, 15.0%)

+ Heterocycles & nucleotides (180 samples, 21.3%)

+ Lipids (189 samples, 22.3%)

+ Nitrogenous & organic acids (251 samples, 29.6%)

+ Organic oxygen compounds (100 samples, 11.8%)

*Final dataset size:* 847 compounds across 5 classes.

The class distribution shows moderate imbalance, with the largest class (Nitrogenous & organic acids) containing 2.5#sym.times more samples than the smallest class (Organic oxygen compounds). This imbalance was addressed during model training through stratified sampling and class weighting. // ?

== Preprocessing

=== Peak List Parsing
Raw HNMR peak lists were parsed to extract chemical shift (ppm) and intensity information. Files contained three sections:

- *Table of Peaks:* Primary data with ppm values and measured heights

- *Table of Assignments:* Peak assignments to specific atoms

- *Table of Multiplets:* Grouped peaks with coupling information

Peak extraction prioritized the "Table of Peaks" section, using measured heights as intensity values. Only peaks within the 0-10 ppm range were retained, as this encompasses the typical HNMR chemical shift range while excluding artifacts.

=== Spectral Binning
To convert variable-length peak lists into fixed-dimension feature vectors suitable for machine learning, spectra were binned using the following parameters:

- *ppm range:* 0-10 ppm

- *Number of bins:* 400

- *Resolution:* 0.025 ppm per bin

- *Binning method:* Maximum intensity pooling (when multiple peaks fell within the same bin, the maximum intensity was retained)

The binning resolution of 0.025 ppm was chosen to balance spectral detail with computational efficiency. This resolution is sufficient to distinguish most chemically meaningful peak differences while avoiding excessive sparsity.

=== Normalization
A multi-step normalization procedure was applied to each binned spectrum:

+ *Log transformation:* Applied log(1 + x) to compress the dynamic range of intensities, which can span several orders of magnitude

+ *Per-spectrum max normalization:* Each spectrum was scaled by dividing by its maximum bin value, normalizing to the [0, 1] range

+ *Z-score normalization:* Features were standardized using mean and standard deviation calculated from the training set only, preventing data leakage

This normalization pipeline ensures that the model learns from spectral patterns (peak positions and relative intensities) rather than absolute intensity scales, which can vary due to instrumental factors.

== Data Splits
Data were partitioned into training, validation, and test sets using stratified random sampling to preserve class distributions:

- *Training set:* 592 samples (69.9%)

- *Validation set:* 127 samples (15.0%)

- *Test set:* 128 samples (15.1%)

Stratification ensured each split maintained approximately the same class proportions as the full dataset. A fixed random seed (42) was used for reproducibility. The validation set was used for hyperparameter tuning and early stopping, while the test set remained completely held-out until final model evaluation.

= Method Description
// workflow, models, evaluation metrics.

= Results
// figures, tables, performance metrics vs. baseline.

== CNN

#figure(
  image("/fig/pca_visualization.png"),
  caption: [PCA visualization]
)

#figure(
  image("/fig/cnn_training_results.png"),
  caption: [CNN training results.]
)

#figure(
  image("/fig/cnn_simplified_results.png"),
  caption: [CNN simplified results.]
)

== Random Forest

#figure(
  image("/fig/rf_tuned_results.png"),
  caption: [RF tuned results]
)

= Conclusion & Discussion
// findings, limitations, future directions.

= Data and Code Availability
// links to dataset and repo (per FAIR guidelines).

= Acknowledgments
// contributions, support, and note on GenAI tools used.

// References
// relevant literature.
#bibliography("references.bib", style: "ieee", title: "References")

= Appendices
// AI deep research log (mandatory), prompts, agent transcripts, extra figures.