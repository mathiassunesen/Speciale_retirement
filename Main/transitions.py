import numpy as np
from numba import njit, prange

@njit(parallel=True)
def age(t): 
    return t+57

@njit(parallel=True)
def oap(t,par): # folkepension
    return 61152

@njit(parallel=True)
def pension(t,par): # efterløn
    if age(t) < 60:
        p = 0
    elif (60 <= age(t) < 62):
        p = 166400
    elif (62 <= age(t) < 65):
        p = 182780
    else:
        p = oap(t,par)
    return p/100000
    #return 0

@njit(parallel=True)
def income(t,par):
    return (-5.999 + 0.629*age(t) - 0.532*age(t)**2/100)/5.5

@njit(parallel=True)
def survival(t,par):
    return par.survival_probs[t]
