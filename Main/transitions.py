import numpy as np
from numba import njit, prange

@njit(parallel=True)
def age(t): 
    return t+57

@njit(parallel=True)
def oap(t): # folkepension
    return 61152 + 61560

@njit(parallel=True)
def pension(t,a): # efterløn

    p = np.zeros_like(a)

    if age(t) < 60:
        p[:] = 0

    elif (60 <= age(t) < 62):
        priv = priv_pension(t,a)
        for i in prange(len(priv)):
            p[i] = max(0,166400 - 0.6*0.05*max(0,priv[i] - 12600))
    
    elif (62 <= age(t) < 65):
        p[:] = 182780
        
    else:
        p[:] = oap(t)
    
    return p/100000
    #return 0

@njit(parallel=True)
def priv_pension(t,a):
    at = age(t)
    lw = np.log(a)
    frac = -57.670 + 0.216*at - 0.187*(at**2)/100 + 12.057*lw -0.920*(lw**2) + 0.023*(lw**3)
    return frac*a

@njit(parallel=True)
def labor_income(t,par):
    return np.exp(-15.956 + 0.934*age(t) - 0.770*age(t)**2/100)/100000

@njit(parallel=True)
def income(t,par):
    inc = labor_income(t,par)
    personal_income = (1 - par.tau_LMC)*inc
    taxable_income = personal_income - min(par.WD*inc,par.WD_upper)
    y_low_l = par.y_low + max(0,par.y_low-0)
    T_c = max(0,par.tau_c*(taxable_income - y_low_l))
    T_h = max(0,par.tau_h*(taxable_income - y_low_l))
    T_l = max(0,par.tau_m*(personal_income - y_low_l))
    T_m = max(0,par.tau_m*(personal_income - par.y_low_m))
    T_u = max(0,min(par.tau_u,par.tau_max)*(personal_income - par.y_low_u))
    return personal_income - T_c - T_h - T_l - T_m - T_u

@njit(parallel=True)
def survival(t,par):
    #return par.survival_probs[t]
    tmp = min(1,np.exp(-10.338 + 0.097*age(t)))
    return min(1,(1 - tmp))
