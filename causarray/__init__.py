import os
if 'NUMBA_THREADING_LAYER' not in os.environ:
    # Prefer OpenMP (fork-safe with llvm-openmp) over TBB (not fork-safe).
    # Falls back to workqueue (serial but safe) if OMP is unavailable.
    # TBB is listed last because it causes warnings when joblib forks.
    os.environ['NUMBA_THREADING_LAYER_PRIORITY'] = 'omp workqueue tbb'
os.environ.setdefault('KMP_WARNINGS', '0')

__all__ = [
    'LFC', 'gcate_lfc_batch', 'LFC_batch',
    'fit_glm', 'fit_glm_fast', 'fit_glm_ondisk',
    'reset_random_seeds', 'fit_gcate', 'fit_gcate_batch',
    ]


from causarray.DR_learner import LFC, gcate_lfc_batch, LFC_batch  # ATE, SATE, FC
from causarray.gcate_glm import fit_glm
from causarray.nb_glm_fast import fit_glm_fast, fit_glm_ondisk
from causarray.utils import prep_causarray_data, reset_random_seeds, comp_size_factor

from causarray.gcate import *
from causarray.__about__ import __version__

__license__ = "MIT"

__author__ = "Jin-Hong Du, Maya Shen, Hansruedi Mathys, and Kathryn Roeder"
__maintainer__ = "Jin-Hong Du"
__maintainer_email__ = "jinhongd@andrew.cmu.edu"
__description__ = ("Causarray: A Python package for simultaneous causal inference"
    " with an array of outcomes."
    )