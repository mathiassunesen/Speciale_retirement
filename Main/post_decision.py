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
    Y = transitions.income(t,par)
    m_plus = par.R*a + Y

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
def compute(t,sol,par):
    """ compute the post-decision functions """

    # unpack (helps numba optimize)
    c = sol.c[t+1]
    m = sol.m[t+1]
    v = sol.v[t+1]
    
    c_plus_interp = sol.c_plus_interp[t]
    v_plus_interp = sol.v_plus_interp[t]
    c_plus_retired_interp = sol.c_plus_retired_interp[t]
    v_plus_retired_interp = sol.v_plus_retired_interp[t]  
    q = sol.q[t]
    v_plus_raw = sol.v_plus_raw[t]

    # retired - no uncertainty
    # a. next period ressources and value
    w = par.xi_w
    a = par.grid_a
    m_plus = par.R*a

    # b. interpolate   
    #prep_c = linear_interp.interp_1d_prep(len(m_plus))
    #linear_interp.interp_1d_vec_mon_rep(prep_c,m[:,0],c[:,0],m_plus,c_plus_retired_interp)    
    #prep_v = linear_interp.interp_1d_prep(len(m_plus))
    #linear_interp.interp_1d_vec_mon_rep(prep_v,m[:,0],c[:,0],m_plus,c_plus_retired_interp)    
    linear_interp.interp_1d_vec(m[:,0],c[:,0],m_plus,c_plus_retired_interp)
    linear_interp.interp_1d_vec(m[:,0],v[:,0],m_plus,v_plus_retired_interp)     

    # c. next period marginal utility
    marg_u_plus = utility.marg_func(c_plus_retired_interp[:],par)

    # d. store results
    v_plus_raw[:,0] = v_plus_retired_interp
    q[:,0] = par.beta*par.R*marg_u_plus

    # working
    # a. next period ressources and value
    w = par.xi_w
    a = par.a_work
    xi = par.xi_work
    m_plus = par.R*a + par.W*xi
    m_plus_long = m_plus.reshape(m_plus.size,1) # reshape to one long vector

    # b. interpolate
    for iz in prange(2): # in parallel
        
        #prep_c = linear_interp.interp_1d_prep(len(m_plus_long))
        #prep_v = linear_interp.interp_1d_prep(len(m_plus_long))
        #linear_interp.interp_1d_vec_mon_rep(prep_c,m[:,iz],c[:,iz],m_plus_long[:,0],c_plus_interp[:,iz])
        #linear_interp.interp_1d_vec_mon_rep(prep_v,m[:,iz],v[:,iz],m_plus_long[:,0],v_plus_interp[:,iz])        
        linear_interp.interp_1d_vec(m[:,iz],c[:,iz],m_plus_long[:,0],c_plus_interp[:,iz])
        linear_interp.interp_1d_vec(m[:,iz],v[:,iz],m_plus_long[:,0],v_plus_interp[:,iz])

        #c_interp = RegularGridInterpolator([m[:,iz]], c[:,iz],method='linear',bounds_error=False,fill_value=None)
        #v_interp = RegularGridInterpolator([m[:,iz]], v[:,iz],method='linear',bounds_error=False,fill_value=None)
        #c_plus_interp[:,iz] = c_interp(m_plus_long)
        #v_plus_interp[:,iz] = v_interp(m_plus_long)

    # c. continuation value - integrate out taste and income shock
    logsum,prob = funs.logsum_vec(v_plus_interp[:,:],par.sigma_eta) # taste shock
    logsum = logsum.reshape(m_plus.shape) # reshape
    prob = prob[:,0].reshape(m_plus.shape)
    vplus_raw = np.sum(w*logsum, axis=1) # income shock

    # d. reshape
    c_plus0 = c_plus_interp[:,0].reshape(m_plus.shape)
    c_plus1 = c_plus_interp[:,1].reshape(m_plus.shape)

    # d. expected future marginal utility - integrate out taste and income shock
    marg_u_plus = prob*utility.marg_func(c_plus0,par) + (1-prob)*utility.marg_func(c_plus1,par)
    avg_marg_u_plus = np.sum(w*marg_u_plus, axis=1) # income shock

    # e. store result in q
    v_plus_raw[:,1] = vplus_raw
    q[:,1] = avg_marg_u_plus