import os
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)