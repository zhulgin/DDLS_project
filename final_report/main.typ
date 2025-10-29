#let title = "Automated Metabolite Family Identification from Proton NMR Data"

#set text(10pt, font: "New Computer Modern")
#show link: set text(fill: blue)
#set document(title: title)
#set page(
  header: text[
    #set align(right)
    #set text(9pt)
    _#[#title]_\
    DDLS 2025\
    Alfred Larsson
  ],
  numbering: "1"
)

#show "HNMR": [#super[1]H NMR]
#let rawbox(body) = box(fill: gray.lighten(85%), inset: 8pt, radius: 6pt)[#body]
#show figure.where(kind: "code"): set figure(supplement: "Code")
#show figure.caption: set text(10pt, style: "italic")
#show "CH2": [CH#sub[2]]

#outline()

#pagebreak(weak: true)

= Abstract
// (≤100 words): problem, method, results.
#include("abstract.typ")

#pagebreak(weak: true)

= Background
// Background and Motivation: why you chose this dataset/task.
#include("background.typ")

= Dataset Summary
// data source, preprocessing, splits, distributions.
#include("dataset_summary.typ")

= Method Description
// workflow, models, evaluation metrics.
#include("method.typ")

= Results
// figures, tables, performance metrics vs. baseline.
#include("results.typ")





= Conclusion & Discussion
// findings, limitations, future directions.
#include("conclusion.typ")

= Data and Code Availability
// links to dataset and repo (per FAIR guidelines).
All code and figures are publicly available in the #link("https://github.com/zhulgin/DDLS_project")[Github repo.]

= Acknowledgments
// contributions, support, and note on GenAI tools used.
ChatGPT and Claude were used for assistance with this project. The author thanks the DDLS course teachers for an informative and fun course.

// References
// relevant literature.
#bibliography("references.bib", style: "ieee", title: "References")

#pagebreak(weak: true)
= Appendices
// AI deep research log (mandatory), prompts, agent transcripts, extra figures.

#include("appendix_figures.typ")

== AI Transcripts

#link("https://chatgpt.com/g/g-p-68dcef6aa5888191b9244b9c649a83d7-ddls-project/project")[Link to ChatGPT transcripts.]

#link("https://claude.ai/share/540e0e6c-0ca4-468d-9bea-bb139aeb1cd7")[Link to Claude transcripts.]

== Deep Research Report