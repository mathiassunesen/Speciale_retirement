import numpy as np
from numba import njit, prange

@njit(parallel=True)
def age(t,par): 
    return t+par.start_T

@njit(parallel=True)
def inv_age(age,par): 
    return age-par.start_T


@njit(parallel=True)
def state_translate(st,ind,par):
    if par.states[st][par.var[ind]] == 1:
        return 1
    else:
        return 0    

@njit(parallel=True)
def oap(t,par): # folkepension
    return par.oap_base + par.oap_add 

@njit(parallel=True)
def pension(t,st,a,retirement,par):

    pens = np.zeros(a.size)

    if age(t,par) >= 65:
        pens[:] = oap(t,par)
    
    else:
        if retirement == 0: # two year rule
            pens[:] = par.erp_high  
        elif retirement == 1: # no two year rule
            priv = priv_pension(t,st,a,par)
            for i in range(len(priv)):
                pens[i] = max(0,166400 - 0.6*0.05*max(0,priv[i] - 12600))
        elif retirement == 2: # no erp
            pass        
    
    return pens/100000


@njit(parallel=True)
def priv_pension(t,st,a,par):

    priv = np.zeros(a.size)
    if a.size == 1: # convert to array - to be compatible with simulate, which is loop based
        a = np.array([a])
        
    ag = age(t,par)
    hs = state_translate(st,'high_skilled',par)
    ch = state_translate(st,'children',par)
    
    for i in range(len(priv)):
        if a[i] <= par.tol:
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
def income(t,st,par,shock):
    inc = par.labor_inc_array[t,st]*shock # look up
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
def labor_income_fill_out(par):
    num_st = len(par.states)
    labor_inc_array = np.zeros((par.T,num_st))
    for t in range(par.T):
        for st in range(num_st):
            labor_inc_array[t,st] = labor_income(t,st,par)
    return labor_inc_array


@njit(parallel=True)
def labor_income(t,st,par):
    ag = age(t,par)
    hs = state_translate(st,'high_skilled',par)
    ch = state_translate(st,'children',par)    

    if state_translate(st,'male',par) == 1:
        rlm = par.reg_labor_male
        log_inc = rlm['cons'] + rlm['high_skilled']*hs + rlm['age']*ag + rlm['age2']*(ag**2)/100 + rlm['children']*ch
    else:
        rlf = par.reg_labor_female
        log_inc = rlf['cons'] + rlf['high_skilled']*hs + rlf['age']*ag + rlf['age2']*(ag**2)/100 + rlf['children']*ch
    return np.exp(log_inc)/100000


@njit(parallel=True)
def survival_look_up(t,st,par):
    if state_translate(st,'male',par) == 1:
        return par.survival_array[t,1]
    else:
        return par.survival_array[t,0]


@njit(parallel=True)
def survival_fill_out(par):
    survival_array = np.zeros((par.T,2)) # only depends age and gender
    for t in range(par.T):
        for st in range(2): # first is women and last is men
            survival_array[t,st] = survival(t,st,par)
    return survival_array


@njit(parallel=True)
def survival(t,st,par):
    ag = age(t,par)

    if st == 1:
        rsm = par.reg_survival_male        
        tmp = min(1,np.exp(rsm['cons'] + rsm['age']*ag))
    else:
        rsf = par.reg_survival_female            
        tmp = min(1,np.exp(rsf['cons'] + rsf['age']*ag))
    return min(1,(1 - tmp))