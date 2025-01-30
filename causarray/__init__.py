__all__ = [
    'LFC', #'ATE', 'SATE', 'FC', 
    'fit_glm', 'reset_random_seeds', 'fit_gcate'
    ]


from causarray.DR_learner import LFC # ATE, SATE, FC
from causarray.gcate_glm import fit_glm
from causarray.utils import prep_causarray_data, reset_random_seeds, comp_size_factor

from causarray.gcate import *
from causarray._version import __version__

__version__ = '.'.join(__version__.split('.')[:3])
__license__ = "MIT"

__author__ = "Jin-Hong Du, Maya Shen, Hansruedi Mathys, and Kathryn Roeder"
__maintainer__ = "Jin-Hong Du"
__maintainer_email__ = "jinhongd@andrew.cmu.edu"
__description__ = ("Causarray: A Python package for simultaneous causal inference"
    " with an array of outcomes."
    )