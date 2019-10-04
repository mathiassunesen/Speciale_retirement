# global modules
from numba import njit, prange
import numpy as np

# local modules
import utility

@njit(parallel=True)
def solve(t,ad,ma,st,ra,d,sol,par):
    """ solve the problem for singles in the last period"""
        
    # unpack (helps numba optimize)
    c = sol.c[t,ad,ma,st,ra,d,:] # time, age difference, gender, states, retirement status, choices, grid
    m = sol.m[t,ad,ma,st,ra,d,:]
    v = sol.v[t,ad,ma,st,ra,d,:]

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
        v[i] = utility.func(c[i],d,ma,st,par) + par.beta*par.gamma*(m[i]-c[i])


@njit(parallel=True)
def solve_c(t,ad,st_h,st_w,erp,d,sol,par):
    """ solve the problem for couples in the last period"""
        
    # unpack (helps numba optimize)
    ad_idx = ad+par.ad_min
    c = sol.c[t,ad_idx,st_h,st_w,erp,d,:] # time, age difference, states (husband), states (wife), erp status, choices, grid
    m = sol.m[t,ad_idx,st_h,st_w,erp,d,:]
    v = sol.v[t,ad_idx,st_h,st_w,erp,d,:]

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
        v[i] = utility.func_c(c[i],d,st_h,st_w,par) + par.beta*par.gamma*(m[i]-c[i])

    

