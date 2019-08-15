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
def priv_pension(t,sol,par):
    at = age(t)
    lw = np.log(sol.a[t-1])
    frac = -57.670 + 0.216*at - 0.187*(at**2)/100 + 12.057*lw -0.920*(lw**2) + 0.023*(lw**3)
    return frac*sol.a[t-1]

@njit(parallel=True)
def income(t,par):
    return np.exp(-15.956 + 0.934*age(t) - 0.770*age(t)**2/100)/100000

@njit(parallel=True)
def survival(t,par):
    #return par.survival_probs[t]
    tmp = min(1,np.exp(-10.338 + 0.097*age(t)))
    return min(1,(1 - tmp))
