.. causarray documentation master file, created by
   sphinx-quickstart on Mon Jan 13 17:38:13 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
=======================


Advances in single-cell sequencing and CRISPR technologies have enabled detailed case-control comparisons and experimental perturbations at single-cell resolution. However, uncovering causal relationships in observational genomic data remains challenging due to selection bias and inadequate adjustment for unmeasured confounders, particularly in heterogeneous datasets. To address these challenges, we introduce `causarray`, a doubly robust causal inference framework for analyzing array-based genomic data at both bulk-cell and single-cell levels. `causarray` integrates a generalized confounder adjustment method to account for unmeasured confounders and employs semiparametric inference with flexible machine learning techniques to ensure robust statistical estimation of treatment effects.


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Main functions

   main_function/gcate
   main_function/lfc


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Tutorials (Python)

   tutorial/perturbseq-py.ipynb

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Tutorials (R)

   tutorial/perturbseq-r.md   
