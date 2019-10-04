# global modules
import numpy as np
from numba import njit, prange



##################################
####      help functions     #####
##################################

@njit(parallel=True)
def age(t,par): 
    """ returns age from t"""
    return t+par.start_T

@njit(parallel=True)
def inv_age(age,par): 
    """ returns t from age"""
    return age-par.start_T

@njit(parallel=True)
def state_translate(st,ind,par):
    """ translates the index st to information about the state

        Args:

            st=state index, ind=str, par=parameters"""

    states = par.states[st]
    if ind == 'male':
        if states[0] == 1:
            return 1
        else:
            return 0
    elif ind == 'elig':
        if states[1] == 1:
            return 1
        else:
            return 0
    elif ind == 'high_skilled':
        if states[2] == 1:
            return 1
        else:
            return 0



##################################
####      pension system     #####
##################################

@njit(parallel=True)
def pension(t,st,a,retirement,par):
    """ returns the pension payment

        Args:

            t=time, st=state
            a=assets/wealth
            retirement=list with info about erp status
            par=parameters"""

    # a. initialize
    pens = np.zeros(a.size)

    # b. oap
    if age(t,par) >= 65:
        pens[:] = oap(t,par)
    
    # c. erp
    elif 60 <= age(t,par) < 65:

        # two year rule
        if retirement == 0:
            pens[:] = par.erp_high  
        
        # no two year rule
        elif retirement == 1:
            priv = priv_pension(t,st,a,par)
            pens[:] = np.maximum(0,166400 - 0.6*0.05*np.maximum(0,priv - 12600))
        
        # no erp
        elif retirement == 2:
            pass       

    # d. pre erp
    else:
        pass 
    
    # d. return (denominated in 100.000 DKR)
    return pens/100000


@njit(parallel=True)
def oap(t,par): 
    """ old age pension (folkepension)"""
    return par.oap_base + par.oap_add 

@njit(parallel=True)
def priv_pension(t,st,a,par):
    """ compute private pension wealth as a function of total wealth (a)

        Args:

            t=time, st=state
            a=total wealth, par=parameters"""

    # a. initialize
    priv = np.zeros(a.size)
    mask_a = np.where(a > par.tol)[0]

    # b. find frac     
    ag = age(t,par)
    hs = state_translate(st,'high_skilled',par)
    lw = np.log(a[mask_a]) # to avoid log of zero!
    if state_translate(st,'male',par) == 1:            
        frac = -57.670 + 0.216*ag - 0.187*(ag**2)/100 + 0.142*hs + 12.057*lw -0.920*(lw**2) + 0.023*(lw**3)
    else:
        frac = -47.565 + 0.098*ag - 0.091*(ag**2)/100 + 0.185*hs + 10.062*lw -0.732*(lw**2) + 0.018*(lw**3)

    # c. calculate priv
    priv[mask_a] = frac*a[mask_a]
    return priv    


# @njit(parallel=True)
# def priv_pension(t,st,a,par):
#     """ compute private pension wealth as a function of total wealth (a)

#         Args:

#             t=time, st=state
#             a=total wealth, par=parameters"""

#     # a. initialize
#     priv = np.zeros(a.size)     
#     ag = age(t,par)
#     hs = state_translate(st,'high_skilled',par)
    
#     # b. loop over a
#     for i in range(len(priv)):
#         if a[i] <= par.tol: # to avoid log of zero!
#             pass
#         else:       
#             lw = np.log(a[i])
#             if state_translate(st,'male',par) == 1:            
#                 frac = -57.670 + 0.216*ag - 0.187*(ag**2)/100 + 0.142*hs + 12.057*lw -0.920*(lw**2) + 0.023*(lw**3)
#             else:
#                 frac = -47.565 + 0.098*ag - 0.091*(ag**2)/100 + 0.185*hs + 10.062*lw -0.732*(lw**2) + 0.018*(lw**3)
#             priv[i] = frac*a[i]
        
#     return priv    



##################################
####      income and tax     #####
##################################

@njit(parallel=True)
def income(t,st,par,shock):
    """ compute after tax labor income

        Args:
            
            t=time, st=state, par=parameters, shock=lognormal shock"""
    
    # a. pre tax labor income
    inc = par.labor_inc_array[t,st]*shock # look up
    
    # b. income definitions
    personal_income = (1 - par.tau_LMC)*inc
    taxable_income = personal_income - np.minimum(par.WD*inc,par.WD_upper)
    y_low_l = par.y_low + np.maximum(0,par.y_low-0)

    # c. taxes
    T_c = np.maximum(0,par.tau_c*(taxable_income - y_low_l))
    T_h = np.maximum(0,par.tau_h*(taxable_income - y_low_l))
    T_l = np.maximum(0,par.tau_m*(personal_income - y_low_l))
    T_m = np.maximum(0,par.tau_m*(personal_income - par.y_low_m))
    T_u = np.maximum(0,np.minimum(par.tau_u,par.tau_max)*(personal_income - par.y_low_u))
    
    # d. return after tax labor income
    return personal_income - T_c - T_h - T_l - T_m - T_u

@njit(parallel=True)
def labor_income_fill_out(par):
    """ precomputes pre tax labor income (absence of shocks)"""
    num_st = len(par.states)
    labor_inc_array = np.zeros((par.T,num_st))
    for t in range(par.T):
        for st in range(num_st):
            labor_inc_array[t,st] = labor_income(t,st,par)
    return labor_inc_array

@njit(parallel=True)
def labor_income(t,st,par):
    """ pre tax labor income for given time and state"""
    ag = age(t,par)
    hs = state_translate(st,'high_skilled',par)

    if state_translate(st,'male',par) == 1:
        log_inc = -15.956 + 0.230*hs + 0.934*ag - 0.770*(ag**2)/100
    else:
        log_inc = -18.937 + 0.248*hs + 1.036*ag - 0.856*(ag**2)/100
    return np.exp(log_inc)/100000



##################################
####         survival        #####
##################################

@njit(parallel=True)
def survival_look_up(t,st,par):
    """ look up in the precomputed survival probabilities"""
    if state_translate(st,'male',par) == 1:
        return par.survival_array[t,1]
    else:
        return par.survival_array[t,0]


@njit(parallel=True)
def survival_fill_out(par):
    """ precomputes survival probabilities across age and state"""
    survival_array = np.zeros((par.T,2))    # only depends on age and gender
    for t in range(par.T):
        for st in range(2):                 # first is women and last is men
            survival_array[t,st] = survival(t,st,par)
    return survival_array

@njit(parallel=True)
def survival(t,st,par):
    """ survival probability for given time and state"""
    ag = age(t,par)

    if st == 1: # male 
        tmp = min(1,np.exp(-10.338 + 0.097*ag))
    else:       # female       
        tmp = min(1,np.exp(-11.142 + 0.103*ag))
    return min(1,(1 - tmp)) 