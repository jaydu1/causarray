Doubly-robust semiparametric inference
======================================

:func:`LFC` estimates per-gene log-fold changes using a doubly-robust AIPW
estimator.  It requires the augmented covariate matrix ``W = [X | U]`` where
``U`` are the latent factors from :func:`fit_gcate`, and produces a DataFrame
with effect estimates, standard errors, and BH-adjusted p-values.

For screens with hundreds of perturbations use :func:`gcate_lfc_batch`, which
runs GCATE and LFC in batches to keep peak memory bounded.

.. automodule:: causarray.DR_learner
   :members:
