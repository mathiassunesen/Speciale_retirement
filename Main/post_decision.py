import numpy as np
from numba import njit, prange
import time
from scipy.interpolate import RegularGridInterpolator

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import funs
import transitions

@njit(parallel=True)
def compute_retired(t,st,sol,par,retirement):
    """compute the post-decision function if retired"""

    # unpack (helps numba optimize)
    c = sol.c[t+1,st,:,0,retirement[0]] # the last dimension indicates main solution(=0) or ad hoc/extra solution
    m = sol.m[t+1,st,:,0,retirement[0]]
    v = sol.v[t+1,st,:,0,retirement[0]]
    
    c_plus_interp = sol.c_plus_interp[t,st,:,0,retirement[2]]
    v_plus_interp = sol.v_plus_interp[t,st,:,0,retirement[2]]  
    q = sol.q[t,st,:,0,retirement[2]]
    v_plus_raw = sol.v_plus_raw[t,st,:,0,retirement[2]]
    
    # a. next period ressources and value
    a = par.grid_a
    m_plus = par.R*a + transitions.pension(t+1,st,a,retirement[1],par)        

    # b. interpolate       
    linear_interp.interp_1d_vec(m,c,m_plus,c_plus_interp)
    linear_interp.interp_1d_vec(m,v,m_plus,v_plus_interp)     

    # c. next period marginal utility
    marg_u_plus = utility.marg_func(c_plus_interp,par)

    # d. store results
    pi = transitions.survival(t,st,par)
    v_plus_raw[:] = v_plus_interp
    q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 


@njit(parallel=True)
def compute_work(t,st,sol,par):
    """compute the post-decision function if working"""

    # unpack (helps numba optimize)
    c = sol.c[t+1,st,:,:,0] # zero in the end extracts the main solution
    m = sol.m[t+1,st,:,:,0]
    v = sol.v[t+1,st,:,:,0]

    if t+1 == par.Tr-2: # if forced to retire next period (which means actual retirement two periods from now due to the timing of retirement decision)
        c[:,1] = c[:,0] # initializing work solution, which is equal to retirement solution
        m[:,1] = m[:,0]
        v[:,1] = v[:,0]
                        
    c_plus_interp = sol.c_plus_interp[t,st,:,:,0]
    v_plus_interp = sol.v_plus_interp[t,st,:,:,0]
    q = sol.q[t,st,:,1,0]
    v_plus_raw = sol.v_plus_raw[t,st,:,1,0]

    # a. next period ressources and value
    Ra = par.R*par.grid_a
    if transitions.state_translate(st,'male',par) == 1:
        w = par.xi_men_w
        xi = par.xi_men
    else:
        w = par.xi_women_w
        xi = par.xi_women
    
    # b. loop over GH nodes
    vp_raw = np.zeros_like(v_plus_raw)
    avg_marg_u_plus = np.zeros_like(q)
    for i in range(len(xi)):
        m_plus = Ra + transitions.income(t+1,st,par,xi[i]) # m_plus is next period resources therefore income(t+1)

        # 1. interpolate and logsum
        for id in prange(2): # in parallel
            linear_interp.interp_1d_vec(m[:,id],c[:,id],m_plus,c_plus_interp[:,id])
            linear_interp.interp_1d_vec(m[:,id],v[:,id],m_plus,v_plus_interp[:,id])

        logsum,prob = funs.logsum_vec(v_plus_interp,par)
        logsum = logsum[:,0]
        prob = prob[:,0]

        # 2. integrate out shocks
        vp_raw += w[i]*logsum # store v_plus_raw
        marg_u_plus = prob*utility.marg_func(c_plus_interp[:,0],par) + (1-prob)*utility.marg_func(c_plus_interp[:,1],par)
        avg_marg_u_plus += w[i]*marg_u_plus

    # c. store q
    pi = transitions.survival(t,st,par)  
    v_plus_raw[:] = vp_raw
    q[:] = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)


@njit(parallel=True)
def value_of_choice_retired(t,st,m,c,sol,par,retirement):
    """compute the value-of-choice of retiring"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros(poc)

    # a. next period ressources
    a = m-c
    m_plus = par.R*a + transitions.pension(t+1,st,a,retirement[1],par)

    # b. next period value
    linear_interp.interp_1d_vec(sol.m[t+1,st,:,0,retirement[0]],sol.v[t+1,st,:,0,retirement[0]],m_plus,v_plus_interp)
    
    # c. value-of-choice
    pi = transitions.survival(t,st,par)
    v = utility.func(c,0,st,par) + par.beta*(pi*v_plus_interp + (1-pi)*par.gamma*a)
    return v

@njit(parallel=True)
def value_of_choice_work(t,st,m,c,sol,par):
    """compute the value-of-choice of working"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros((poc,2))

    # a. next period ressources and value
    a = m - c
    Ra = par.R*a
    if transitions.state_translate(st,'male',par) == 1:
        w = par.xi_men_w
        xi = par.xi_men
    else:
        w = par.xi_women_w
        xi = par.xi_women
    
    # b. loop over GH nodes
    v_plus_raw = np.zeros_like(a)
    for i in range(len(xi)):
        m_plus = Ra + transitions.income(t+1,st,par,xi[i]) # m_plus is next period resources therefore income(t+1)

        # 1. interpolate and logsum
        for id in prange(2): # in parallel
            linear_interp.interp_1d_vec(sol.m[t+1,st,:,id,0],sol.v[t+1,st,:,id,0],m_plus,v_plus_interp[:,id])

        logsum = funs.logsum_vec(v_plus_interp,par)[0]
        logsum = logsum[:,0]

        # 2. integrate out shocks
        v_plus_raw += w[i]*logsum

    # c. value-of-choice
    pi = transitions.survival(t,st,par)    
    v = utility.func(c,1,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*a)
    return v