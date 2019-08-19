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
def compute_retired(t,st,sol,par):
    """compute the post-decision function if retired"""

    # unpack (helps numba optimize)
    c = sol.c[t+1,st,:,0]
    m = sol.m[t+1,st,:,0]
    v = sol.v[t+1,st,:,0]
    
    c_plus_interp = sol.c_plus_interp[t,st,:,0]
    v_plus_interp = sol.v_plus_interp[t,st,:,0]  
    q = sol.q[t,st,:,0]
    v_plus_raw = sol.v_plus_raw[t,st,:,0]
    
    # a. next period ressources and value
    a = par.grid_a
    #m_plus_main = par.R*a + transitions.pension(t,st,a) # if above 61 (two year rule)
    pi = transitions.survival(t,par)

    if transitions.age(t) >= 65:
        m_plus_main = par.R*a + transitions.pension(t,st,a,two_year=1) # oap
        linear_interp.interp_1d_vec(m,c,m_plus_main,c_plus_interp)
        linear_interp.interp_1d_vec(m,v,m_plus_main,v_plus_interp)    
        marg_u_plus = utility.marg_func(c_plus_interp,par) 
        v_plus_raw[:] = v_plus_interp
        q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma)         

    elif 62 <= transitions.age(t) <= 64:
        m_plus_main = par.R*a + transitions.pension(t,st,a,two_year=1) # two-year rule satisfied
        linear_interp.interp_1d_vec(m,c,m_plus_main,c_plus_interp)
        linear_interp.interp_1d_vec(m,v,m_plus_main,v_plus_interp)     
        marg_u_plus = utility.marg_func(c_plus_interp,par) 
        v_plus_raw[:] = v_plus_interp
        q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma)               

        c_60_61 = sol.c_60_61[t+1,st,:,0]
        m_60_61 = sol.m_60_61[t+1,st,:,0]
        v_60_61 = sol.v_60_61[t+1,st,:,0]
        c_plus_interp_60_61 = sol.c_plus_interp_60_61[t,st,:,0]
        v_plus_interp_60_61 = sol.v_plus_interp_60_61[t,st,:,0]  
        q_60_61 = sol.q_60_61[t,st,:,0]
        v_plus_raw_60_61 = sol.v_plus_raw_60_61[t,st,:,0]        
        m_plus_60_61 = par.R*a + transitions.pension(t,st,a,two_year=0) # two-year rule not satisfied
        linear_interp.interp_1d_vec(m_60_61,c_60_61,m_plus_60_61,c_plus_interp_60_61)
        linear_interp.interp_1d_vec(m_60_61,v_60_61,m_plus_60_61,v_plus_interp_60_61)             
        marg_u_plus_60_61 = utility.marg_func(c_plus_interp_60_61,par)
        v_plus_raw_60_61[:] = v_plus_interp_60_61
        q_60_61[:] = par.beta*(par.R*pi*marg_u_plus_60_61 + (1-pi)*par.gamma)                
        
        c_below60 = sol.c_below60[t+1,st,:,0]
        m_below60 = sol.m_below60[t+1,st,:,0]
        v_below60 = sol.v_below60[t+1,st,:,0]
        c_plus_interp_below60 = sol.c_plus_interp_below60[t,st,:,0]
        v_plus_interp_below60 = sol.v_plus_interp_below60[t,st,:,0]  
        q_below60 = sol.q_below60[t,st,:,0]
        v_plus_raw_below60 = sol.v_plus_raw_below60[t,st,:,0]        
        m_plus_below60 = par.R*a # no erp
        linear_interp.interp_1d_vec(m_below60,c_below60,m_plus_below60,c_plus_interp_below60)
        linear_interp.interp_1d_vec(m_below60,v_below60,m_plus_below60,v_plus_interp_below60)
        marg_u_plus_below60 = utility.marg_func(c_plus_interp_below60,par)                     
        v_plus_raw_below60[:] = v_plus_interp_below60
        q_below60[:] = par.beta*(par.R*pi*marg_u_plus_below60 + (1-pi)*par.gamma)        

    elif 60 <= transitions.age(t) <= 61:
        m_plus_main = par.R*a + transitions.pension(t,st,a,two_year=0) # two-year rule not satisfied
        linear_interp.interp_1d_vec(m,c,m_plus_main,c_plus_interp)
        linear_interp.interp_1d_vec(m,v,m_plus_main,v_plus_interp) 
        marg_u_plus = utility.marg_func(c_plus_interp,par)   
        v_plus_raw[:] = v_plus_interp
        q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma)                 

        c_below60 = sol.c_below60[t+1,st,:,0]
        m_below60 = sol.m_below60[t+1,st,:,0]
        v_below60 = sol.v_below60[t+1,st,:,0]
        c_plus_interp_below60 = sol.c_plus_interp_below60[t,st,:,0]
        v_plus_interp_below60 = sol.v_plus_interp_below60[t,st,:,0]  
        q_below60 = sol.q_below60[t,st,:,0]
        v_plus_raw_below60 = sol.v_plus_raw_below60[t,st,:,0]
        m_plus_below60 = par.R*a # no erp
        linear_interp.interp_1d_vec(m_below60,c_below60,m_plus_below60,c_plus_interp_below60)
        linear_interp.interp_1d_vec(m_below60,v_below60,m_plus_below60,v_plus_interp_below60)
        marg_u_plus_below60 = utility.marg_func(c_plus_interp_below60,par)  
        v_plus_raw_below60[:] = v_plus_interp_below60
        q_below60[:] = par.beta*(par.R*pi*marg_u_plus_below60 + (1-pi)*par.gamma)                           

    else:
        m_plus_main = par.R*a # no erp                        
        linear_interp.interp_1d_vec(m,c,m_plus_main,c_plus_interp)
        linear_interp.interp_1d_vec(m,v,m_plus_main,v_plus_interp)             
        marg_u_plus = utility.marg_func(c_plus_interp,par)   
        v_plus_raw[:] = v_plus_interp
        q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma)             

    # b. interpolate       
    #linear_interp.interp_1d_vec(m,c,m_plus_main,c_plus_interp)
    #linear_interp.interp_1d_vec(m,v,m_plus_main,v_plus_interp)     

    # c. next period marginal utility
    #marg_u_plus = utility.marg_func(c_plus_interp,par)

    # d. store results
    #pi = transitions.survival(t,par)
    #v_plus_raw[:] = v_plus_interp
    #q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 


@njit(parallel=True)
def compute_work(t,st,sol,par):
    """compute the post-decision function if working"""

    # unpack (helps numba optimize)
    c = sol.c[t+1,st]
    m = sol.m[t+1,st]
    v = sol.v[t+1,st]

    if t == par.Tr-2: # if forced to retire next period
        c[:,1] = c[:,0]
        m[:,1] = m[:,0]
        v[:,1] = v[:,0]
                        
    c_plus_interp = sol.c_plus_interp[t,st]
    v_plus_interp = sol.v_plus_interp[t,st]
    q = sol.q[t,st,:,1]
    v_plus_raw = sol.v_plus_raw[t,st,:,1]

    # a. next period ressources and value
    a = par.grid_a
    m_plus = par.R*a + transitions.income(t,st,par)

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
def value_of_choice_retired(t,st,m,c,sol,par):
    """compute the value-of-choice"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros(poc)

    # a. next period ressources
    a = m-c
    m_plus = par.R*a + transitions.pension(t,st,a,two_year=1)

    # b. next period value
    linear_interp.interp_1d_vec(sol.m[t+1,st,:,0],sol.v[t+1,st,:,0],m_plus,v_plus_interp)
    
    # c. value-of-choice
    pi = transitions.survival(t,par)
    v = utility.func(c,0,st,par) + par.beta*(pi*v_plus_interp + (1-pi)*par.gamma*a)
    return v

@njit(parallel=True)
def value_of_choice_retired_60_61(t,st,m,c,sol,par):
    """compute the value-of-choice"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros(poc)

    # a. next period ressources
    a = m-c
    m_plus = par.R*a + transitions.pension(t,st,a,two_year=0)

    # b. next period value
    linear_interp.interp_1d_vec(sol.m_60_61[t+1,st,:,0],sol.v_60_61[t+1,st,:,0],m_plus,v_plus_interp)
    
    # c. value-of-choice
    pi = transitions.survival(t,par)
    v = utility.func(c,0,st,par) + par.beta*(pi*v_plus_interp + (1-pi)*par.gamma*a)
    return v

@njit(parallel=True)
def value_of_choice_retired_below60(t,st,m,c,sol,par):
    """compute the value-of-choice"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros(poc)

    # a. next period ressources
    a = m-c
    m_plus = par.R*a

    # b. next period value
    linear_interp.interp_1d_vec(sol.m_below60[t+1,st,:,0],sol.v_below60[t+1,st,:,0],m_plus,v_plus_interp)
    
    # c. value-of-choice
    pi = transitions.survival(t,par)
    v = utility.func(c,0,st,par) + par.beta*(pi*v_plus_interp + (1-pi)*par.gamma*a)
    return v    

@njit(parallel=True)
def value_of_choice_work(t,st,m,c,sol,par):
    """compute the value-of-choice"""
    
    # initialize
    poc = par.poc
    v_plus_interp = np.nan*np.zeros((poc,2))

    # a. next period ressources
    a = m-c
    m_plus = par.R*a + transitions.income(t,st,par)

    # b. next period value
    for id in prange(2):
        linear_interp.interp_1d_vec(sol.m[t+1,st,:,id],sol.v[t+1,st,:,id],m_plus,v_plus_interp[:,id])

    logsum = funs.logsum_vec(v_plus_interp,par)[0]
    logsum = logsum.reshape(a.shape)
    
    # c. value-of-choice
    pi = transitions.survival(t,par)
    v = utility.func(c,1,st,par) + par.beta*(pi*logsum + (1-pi)*par.gamma*a)
    return v