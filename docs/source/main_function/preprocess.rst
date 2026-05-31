Preprocessing
=================================

:func:`prep_causarray_data` is the first step in the causarray pipeline.
It validates and formats the count matrix ``Y``, treatment matrix ``A``,
and optional covariate matrices ``X`` / ``X_A`` into the shapes expected
by :func:`fit_gcate` and :func:`LFC`.  An intercept column is added to
``X`` automatically, and a standardised log-library-size covariate is
appended to ``X_A`` for the propensity model.

.. automodule:: causarray.utils
   :members: prep_causarray_data
