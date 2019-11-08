# global modules
from numba import njit, prange
import numpy as np

# local modules
import utility
import transitions

@njit(parallel=True)
def solve(t,ad,ma,st,ra,d,sol_c,sol_m,sol_v,par):
    """ solve the model in the last period for singles """
        
    # unpack (helps numba optimize)
    c = sol_c[t,ra,d,:] # time, age difference, gender, states, retirement status, choices, grid
    m = sol_m[:]
    v = sol_v[t,ra,d,:]

    # initialize
    c[:] = m[:]
    
    # optimal choice
    cons = (par.beta*par.gamma)**(-1/par.rho)
    for i in range(len(m)):
        if m[i] > cons:
            c[i] = cons
        else:
            c[i] = m[i]
        v[i] = utility.func(c[i],d,ma,st,par) + par.beta*par.gamma*(m[i]-c[i])


@njit(parallel=True)
def solve_c(t,ad,st_h,st_w,ra_h,ra_w,d_h,d_w,sol_c,sol_m,sol_v,par):
    """ solve the model in the last period for couples
    
    Args:
        t (int): model time (age of husband)
        ad (int): age difference = age of wife - age of husband
        st_h (int): states of husband
        st_w (int): states of wife
        ra_h (int): retirement status for husband
        ra_w (int): retirement status for wife
        d (int): joint retirement choice
        sol (class): solution
        par (class): parameters

    Returns:
        stores c,m,v in sol
    """
        
    # unpack (helps numba optimize)
    ad_idx = ad+par.ad_min
    d = transitions.d_c(d_h,d_w)
    c = sol_c[t,ad_idx,st_h,st_w,ra_h,ra_w,d,:]
    m = sol_m[t,ad_idx,st_h,st_w,ra_h,ra_w,d,:]
    v = sol_v[t,ad_idx,st_h,st_w,ra_h,ra_w,d,:]

    # initialize
    m[:] = par.grid_a
    c[:] = m[:]
    
    # optimal choice
    cons = (par.beta*par.gamma)**(-1/par.rho)
    for i in range(len(m)):
        if m[i] > cons:
            c[i] = cons
        else:
            c[i] = m[i]
        v[i] = utility.func_c(c[i],d_h,d_w,st_h,st_w,par) + par.beta*par.gamma*(m[i]-c[i])

    

