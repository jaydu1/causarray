Doubly-robust semiparametric inference
======================================

:func:`LFC` estimates per-gene log-fold changes using a doubly-robust AIPW
estimator.  It requires the augmented covariate matrix ``W = [X | U]`` where
``U`` are the latent factors from :func:`fit_gcate`, and produces a DataFrame
with effect estimates, standard errors, and BH-adjusted p-values.

For screens with hundreds of perturbations use :func:`gcate_lfc_batch`, which
runs GCATE and LFC in batches to keep peak memory bounded.

Effect-size columns
-------------------

The returned DataFrame reports the treatment-versus-control effect on both
natural-log and base-2 scales:

=================  ===========================================================
Column             Definition
=================  ===========================================================
``tau``            Natural logarithm of the treated/control mean ratio.
``std``            Standard error of ``tau`` on the natural-log scale.
``log2fc``         Base-2 log fold change, exactly ``tau / log(2)``.
``log2fc_se``      Standard error of ``log2fc``, exactly ``std / log(2)``.
=================  ===========================================================

``log2fc_se`` is the standard error of the estimated log2 fold change, not the
sample standard deviation of expression. The rescaling leaves ``stat``,
p-values, adjusted p-values, and discoveries unchanged. ``tau`` and ``std``
remain available for backward compatibility. The same columns are returned by
``gcate_lfc_batch``; compatible caches made by older versions are upgraded in
memory when loaded.

Choosing the variance estimator
-------------------------------

``LFC`` defaults to ``usevar='unequal'`` (Welch inference), which estimates
the treatment and control variances separately. Prefer this default when the
treatment and control sample sizes, or their propensity-weighted effective
sample sizes, are meaningfully unbalanced. Also retain it when arm-specific
pseudo-outcome variances may differ and for case-control, bulk, and donor-level
pseudo-bulk analyses. Independence of the rows does not imply equal
treatment-arm variances: disease severity, biological response, residual
composition, library size, treatment imbalance, and heterogeneous expression
can all make pooled inference anti-conservative.

For a small, approximately balanced perturbation comparison,
``usevar='pooled'`` may provide better power when the independent sampling
units and arm-specific pseudo-outcome variances are reasonably comparable.
Treat it as an opt-in, empirically justified analysis rather than an automatic
small-sample choice. Balanced counts alone do not justify pooling in a
case-control study. Pooled inference can produce much smaller standard errors
and substantially more discoveries. For a deliberately justified batched
analysis, pass ``lfc_kwargs=dict(usevar='pooled')``.

There is no universal sample-size ratio at which the recommendation changes.
Compare nominal arm sizes, propensity-weighted effective sample sizes,
arm-specific pseudo-outcome variability, and the stability of discoveries
under both estimators. Retain ``usevar='unequal'`` when these diagnostics do not
support pooling.

Donor-level independence alone does not establish equal arm variances. For
example, the SEA-AD tutorial uses ``usevar='unequal'`` because disease severity,
inter-individual response, residual cell composition, and library-size
variation can produce different gene-wise variability between disease groups.

Welch inference does not itself model within-subject correlation. Repeated
cells from the same donor or experimental unit should still be pseudo-bulked
or handled with cluster-aware inference; ``usevar='unequal'`` only protects
against unequal arm variances.

Propensity diagnostics
----------------------

:func:`estimate_propensity_scores` estimates propensity scores without fitting
the outcome model.  Use ``K=5`` to obtain out-of-fold scores for positivity and
overfitting diagnostics.  :func:`summarize_propensity_scores` reports overlap,
tail mass, and inverse-weight effective sample size, while
:func:`plot_propensity_scores` compares treatment and control distributions.

Both the standalone estimator and ``LFC`` use class-balanced logistic
propensity fitting by default. This preserves historical causarray behavior and
ensures that standalone overlap diagnostics describe the same nuisance model
used for effect estimation. Pass ``class_weight=None`` to
``estimate_propensity_scores`` or ``ps_class_weight=None`` to ``LFC`` for a
calibrated-probability sensitivity analysis.

``LFC`` uses the standard AIPW pseudo-outcome, which may be negative for
individual cells even though its counterfactual mean is positive. Individual
pseudo-outcomes are never clipped because doing so can bias the arm means,
particularly when a large shared control group is compared with much smaller
treatment groups.

For a calibrated-propensity sensitivity analysis, use
``LFC(..., ps_class_weight=None)`` and diagnose the matching scores with
``estimate_propensity_scores(..., class_weight=None)``.

The result includes ``mean_control``, ``mean_treated``, and ``estimable``. These
are computed from the unclipped pseudo-outcomes. For numerical stability, valid
aggregate arm means are floored at ``thres_diff`` only when constructing the
log ratio and delta-method denominator. A pair with a nonfinite or nonpositive
raw aggregate mean remains non-estimable, so the aggregate floor cannot create
an extreme discovery from an invalid estimate.

.. automodule:: causarray.DR_learner
   :members:

.. automodule:: causarray.DR_estimation
   :members: estimate_propensity_scores

.. automodule:: causarray.diagnostics
   :members:
