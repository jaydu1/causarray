Generalized confounder estimation
=================================

The GCATE (Generalized Confounder Adjustment for Treatment Effects) functions
estimate latent factors that capture unmeasured confounders in the data.
Call :func:`estimate_r` first to select the number of factors *r* via the JIC
criterion, then call :func:`fit_gcate` (or :func:`fit_gcate_batch` for
large-scale screens) to obtain the latent factor matrix ``U``.  Append ``U``
to the observed covariate matrix before calling :func:`LFC`.

.. automodule:: causarray.gcate
   :members:
