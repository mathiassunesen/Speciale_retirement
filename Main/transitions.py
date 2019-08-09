import numpy as np
from numba import njit, prange

@njit(parallel=True)
def income(t,par):
    return par.Y

@njit(parallel=True)
def survival(t,par):
    return par.survival_probs[t]
