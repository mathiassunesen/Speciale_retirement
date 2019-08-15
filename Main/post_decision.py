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
def compute_retired(t,sol,par):
    """compute the post-decision function if retired"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t+1,:,0] # leave/ignore points on constraint
    m = sol.m[t+1,:,0]
    v = sol.v[t+1,:,0]
    
    c_plus_retired_interp = sol.c_plus_retired_interp[t]
    v_plus_retired_interp = sol.v_plus_retired_interp[t]  
    q = sol.q[t,:,0]
    v_plus_raw = sol.v_plus_raw[t,:,0]
    
    # a. next period ressources and value
    a = par.grid_a
    m_plus = par.R*a + transitions.pension(t,par)

    # b. interpolate       
    linear_interp.interp_1d_vec(m,c,m_plus,c_plus_retired_interp)
    linear_interp.interp_1d_vec(m,v,m_plus,v_plus_retired_interp)     

    # c. next period marginal utility
    marg_u_plus = utility.marg_func(c_plus_retired_interp,par)

    # d. store results
    pi = transitions.survival(t,par)
    v_plus_raw[:] = v_plus_retired_interp
    q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 


@njit(parallel=True)
def compute_work(t,sol,par):
    """compute the post-decision function if working"""

    # unpack (helps numba optimize)
    c = sol.c[t+1]
    m = sol.m[t+1]
    v = sol.v[t+1]

    if t == par.Tr-2: # if forced to retire next period
        c[:,1] = c[:,0]
        m[:,1] = m[:,0]
        v[:,1] = v[:,0]
                        
    c_plus_interp = sol.c_plus_interp[t]
    v_plus_interp = sol.v_plus_interp[t]
    q = sol.q[t,:,1]
    v_plus_raw = sol.v_plus_raw[t,:,1]

    # a. next period ressources and value
    a = par.grid_a
    m_plus = par.R*a + transitions.income(t,par)

    # b. interpolate
    for id in prange(2): # in parallel
        linear_interp.interp_1d_vec(m[:,id],c[:,id],m_plus,c_plus_interp[:,id])
        linear_interp.interp_1d_vec(m[:,id],v[:,id],m_plus,v_plus_interp[:,id])

    # c. continuation value - integrate out taste shock
    logsum,prob = funs.logsum_vec(v_plus_interp,par)
    logsum = logsum.reshape(m_plus.shape)
    prob = prob[:,0].reshape(m_plus.shape)

    # d. reshape
    c_plus0 = c_plus_interp[:,0].reshape(m_plus.shape)
    c_plus1 = c_plus_interp[:,1].reshape(m_plus.shape)

    # d. expected future marginal utility - integrate out taste shock
    marg_u_plus = prob*utility.marg_func(c_plus0,par) + (1-prob)*utility.marg_func(c_plus1,par)

    # e. store result in q
    pi = transitions.survival(t,par)    
    v_plus_raw[:] = logsum
    q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma)


@njit(parallel=True)
def value_of_choice_retired(t,m,c,sol,par):
    """compute the value-of-choice"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros(poc)

    # a. next period ressources
    a = m-c
    m_plus = par.R*a + transitions.pension(t,par)

    # b. next period value
    linear_interp.interp_1d_vec(sol.m[t+1,:,0],sol.v[t+1,:,0],m_plus,v_plus_interp)
    
    # c. value-of-choice
    pi = transitions.survival(t,par)
    v = utility.func(c,0,par) + par.beta*(pi*v_plus_interp + (1-pi)*par.gamma*a)
    return v


@njit(parallel=True)
def value_of_choice_work(t,m,c,sol,par):
    """compute the value-of-choice"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros((poc,2))

    # a. next period ressources
    a = m-c
    m_plus = par.R*a + transitions.income(t,par)

    # b. next period value
    for id in prange(2):
        linear_interp.interp_1d_vec(sol.m[t+1,:,id],sol.v[t+1,:,id],m_plus,v_plus_interp[:,id])

    logsum,prob = funs.logsum_vec(v_plus_interp,par)
    logsum = logsum.reshape(a.shape)
    
    # c. value-of-choice
    pi = transitions.survival(t,par)
    v = utility.func(c,1,par) + par.beta*(pi*logsum + (1-pi)*par.gamma*a)
    return v