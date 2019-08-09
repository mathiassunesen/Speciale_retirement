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
    c = sol.c[t+1]
    m = sol.m[t+1]
    v = sol.v[t+1]
    
    c_plus_retired_interp = sol.c_plus_retired_interp[t]
    v_plus_retired_interp = sol.v_plus_retired_interp[t]  
    q = sol.q[t]
    v_plus_raw = sol.v_plus_raw[t]
    
    # a. next period ressources and value
    a = par.grid_a
    m_plus = par.R*a

    # b. interpolate       
    linear_interp.interp_1d_vec(m[:,0],c[:,0],m_plus,c_plus_retired_interp)
    linear_interp.interp_1d_vec(m[:,0],v[:,0],m_plus,v_plus_retired_interp)     

    # c. next period marginal utility
    marg_u_plus = utility.marg_func(c_plus_retired_interp[:],par)

    # d. store results
    pi = transitions.survival(t,par)
    v_plus_raw[:,0] = v_plus_retired_interp
    q[:,0] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 


@njit(parallel=True)
def compute_work(t,sol,par):
    """compute the post-decision function if working"""

    # unpack (helps numba optimize)
    c = sol.c[t+1]
    m = sol.m[t+1]
    v = sol.v[t+1]
    
    c_plus_interp = sol.c_plus_interp[t]
    v_plus_interp = sol.v_plus_interp[t]
    q = sol.q[t]
    v_plus_raw = sol.v_plus_raw[t]

    # a. next period ressources and value
    a = par.grid_a
    Y = transitions.income(t,par)
    m_plus = par.R*a + Y

    # b. interpolate
    for id in prange(2): # in parallel
        linear_interp.interp_1d_vec(m[:,id],c[:,id],m_plus,c_plus_interp[:,id])
        linear_interp.interp_1d_vec(m[:,id],v[:,id],m_plus,v_plus_interp[:,id])

    # c. continuation value - integrate out taste shock
    logsum,prob = funs.logsum_vec(v_plus_interp[:,:],par.sigma_eta)
    logsum.reshape(m_plus.shape)
    prob = prob[:,0].reshape(m_plus.shape)

    # d. reshape
    c_plus0 = c_plus_interp[:,0].reshape(m_plus.shape)
    c_plus1 = c_plus_interp[:,1].reshape(m_plus.shape)

    # d. expected future marginal utility - integrate out taste shock
    marg_u_plus = prob*utility.marg_func(c_plus0,par) + (1-prob)*utility.marg_func(c_plus1,par)

    # e. store result in q
    pi = transitions.survival(t,par)    
    v_plus_raw[:,1] = logsum
    q[:,1] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma)