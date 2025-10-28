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

HNMR data was retrieved from the Human Metabolome Database (HMDB) @hmdb. The HMDB data is made up of two parts: a #raw(".xml") file containing various information for each compound in the database (HMDB ID, name and synonyms, InChI, description, molecular weight, CAS number, etc.) and a folder containing #raw(".txt") files (see @peaklist) with HNMR peak lists (and in some cases also multiplets and assignments) for a subset of the compounds. 

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

= Method Description
// workflow, models, evaluation metrics.

= Results
// figures, tables, performance metrics vs. baseline.

= Conclusion & Discussion
// findings, limitations, future directions.

= Data and Code Availability
// links to dataset and repo (per FAIR guidelines).

= Acknowledgments
// contributions, support, and note on GenAI tools used.

= References
// relevant literature.

#bibliography("references.bib", style: "ieee")

= Appendices
// AI deep research log (mandatory), prompts, agent transcripts, extra figures.