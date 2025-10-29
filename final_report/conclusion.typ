== Key Findings

This project successfully developed a machine learning pipeline for classifying metabolite compounds based on HNMR spectroscopy data from the Human Metabolome Database. The final Random Forest model achieved 66.4% test accuracy across five compound classes, demonstrating that automated classification of metabolites from spectral data is feasible, though challenging given the inherent complexity and overlap between chemical classes.

== Model Performance and Comparison

/*
*Convolutional Neural Networks (CNN):* Two CNN architectures were tested - a complex model (320 000 parameters) and a simplified version (5157 parameters). Both significantly underperformed, achieving only 15.6% and 21.1% test accuracy respectively, barely above random chance (20%). The CNNs exhibited severe overfitting with validation loss exploding beyond 9000, indicating they struggled to learn generalizable patterns from the sparse, high-dimensional spectral features. *Random Forest (baseline):* Initial implementation with 200 trees and max_depth=20 achieved 64.8% test accuracy, substantially outperforming the CNNs. *Random Forest (tuned):* Systematic hyperparameter optimization via GridSearchCV improved performance to 66.4% test accuracy, representing a modest but meaningful 1.6 percentage point gain over the baseline.

The dramatic superiority of Random Forest over deep learning methods was unexpected but aligns with the data characteristics: with only 847 samples, high feature sparsity, and substantial class overlap (as revealed by PCA), ensemble methods proved more suitable than neural networks requiring large training sets and dense representations.
*/

Two CNN architectures were tested - a complex model (320 000 parameters) and a simplified version (5157 parameters). Both significantly underperformed, achieving only 23.4% and 22.7% test accuracy respectively, barely above random chance (20%). The CNNs exhibited severe overfitting with validation loss exploding beyond 9000, indicating they struggled to learn generalizable patterns from the sparse, high-dimensional spectral features. Initial implementation of Random Forest classifier with 200 trees and max_depth=20 achieved 64.8% test accuracy, substantially outperforming the CNNs. Systematic optimization improved performance to 66.4% test accuracy, representing a 1.6 percentage point gain over the baseline.

The dramatic superiority of Random Forest over deep learning methods was unexpected but aligns with the data characteristics: with only 847 samples, high feature sparsity, and substantial class overlap (as revealed by PCA), ensemble methods proved more suitable than neural networks requiring large training sets and dense representations.


== Model Interpretability

Feature importance analysis revealed that the Random Forest model learned chemically meaningful patterns rather than spurious correlations. *Aliphatic region (1.2-1.6 ppm):* Most discriminative for Lipids, corresponding to methylene groups in fatty acid chains. *Alpha to heteroatom region (3.6 ppm):* Critical for distinguishing Nitrogenous & organic acids and Organic oxygen compounds, where protons adjacent to oxygen or nitrogen atoms resonate. *Aromatic region (6-8 ppm):* Important for Aromatics and Heterocycles & nucleotides, reflecting characteristic aromatic proton chemical shifts.

SHAP analysis demonstrated that individual classification decisions aligned with expected chemical functional groups. For instance, lipid classifications were driven primarily by signals in the 1.3-1.6 ppm range, while aromatic compounds showed strong contributions from the 6.9-7.3 ppm region.

== Limitations

The primary limitation was the modest dataset size of 847 compounds, which proved insufficient for deep learning approaches and limited the model's ability to capture rare spectral patterns. Class imbalance, with the largest class containing 2.5× more samples than the smallest, contributed to lower performance on minority classes despite stratified sampling and class weighting.

The hierarchical grouping from HMDB's original taxonomy into five super-classes involved subjective decisions, introducing classification ambiguity. Some compounds possess characteristics of multiple categories, establishing an inherent ceiling on achievable accuracy even for expert chemists. 

Methodologically, converting variable-length peak lists into fixed 400-bin vectors discarded fine spectral details including peak multiplicity patterns. NMR spectra are also sensitive to experimental conditions (solvent, pH, temperature), introducing variability that complicates pattern recognition. The model's 66.4% accuracy, while substantially better than CNNs, still misclassifies one-third of compounds and lacks explicit uncertainty quantification for flagging unreliable predictions.

== Future directions

Future work should prioritize expanding the dataset through crowdsourcing or database partnerships, potentially reaching thousands of samples to enable more sophisticated modeling approaches. Advanced architectures including graph neural networks (which encode molecular structure directly), hybrid CNN-ensemble methods, and transfer learning from large unlabeled spectral datasets could overcome current performance limitations. The web application could be enhanced with batch processing, confidence thresholding with automatic flagging of uncertain predictions, and integration with laboratory information systems for real-time deployment. 