import numpy as np
from numba import njit, prange

@njit(parallel=True)
def age(t): 
    return t+57

@njit(parallel=True)
def inv_age(age): 
    return age-57


@njit(parallel=True)
def state_translate(st,ind,par):
    if par.states[st][par.var[ind]] == 1:
        return 1
    else:
        return 0    

@njit(parallel=True)
def oap(t): # folkepension
    return 61152 + 61560

@njit(parallel=True)
def pension(t,st,a,retirement,par): # efterløn

    pens = np.zeros_like(a)

    if age(t) >= 65:
        pens[:] = oap(t)
    
    else:
        if retirement == 0: # two year rule
            pens[:] = 182780  
        elif retirement == 1: # no two year rule
            priv = priv_pension(t,st,a,par)
            for i in prange(len(priv)):
                pens[i] = max(0,166400 - 0.6*0.05*max(0,priv[i] - 12600))
        elif retirement == 2: # no erp
            pass        
    
    return pens/100000

@njit(parallel=True)
def priv_pension(t,st,a,par):

    priv = np.zeros_like(a)
    ag = age(t)
    hs = state_translate(st,'high_skilled',par)
    ch = state_translate(st,'children',par)
    
    for i in prange(len(priv)):
        if a[i] <= 1e-6:
            pass
        else:       
            lw = np.log(a[i])
            if state_translate(st,'male',par) == 1:            
                frac = -57.670 + 0.216*ag - 0.187*(ag**2)/100 + 0.142*hs + 0.019*ch + 12.057*lw -0.920*(lw**2) + 0.023*(lw**3)
            else:
                frac = -47.565 + 0.098*ag - 0.091*(ag**2)/100 + 0.185*hs - 0.032*ch + 10.062*lw -0.732*(lw**2) + 0.018*(lw**3)
            priv[i] = frac*a[i]
        
    return priv

@njit(parallel=True)
def labor_income(t,st,par,shock):
    ag = age(t)
    hs = state_translate(st,'high_skilled',par)
    ch = state_translate(st,'children',par)    

    if state_translate(st,'male',par) == 1:
        log_inc = -15.956 + 0.230*hs + 0.934*ag - 0.770*(ag**2)/100 + 0.151*ch
    else:
        log_inc = -18.937 + 0.248*hs + 1.036*ag - 0.856*(ag**2)/100 + 0.021*ch

    return np.exp(log_inc)*shock/100000

@njit(parallel=True)
def income(t,st,par,shock):
    inc = labor_income(t,st,par,shock)
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
def survival(t,st,par):
    ag = age(t)
    if state_translate(st,'male',par) == 1:
        tmp = min(1,np.exp(-10.338 + 0.097*ag))
    else:
        tmp = min(1,np.exp(-11.142 + 0.103*ag))        
    return min(1,(1 - tmp))
