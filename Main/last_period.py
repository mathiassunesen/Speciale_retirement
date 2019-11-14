# global modules
from numba import njit, prange
import numpy as np

# local modules
import utility
import transitions

@njit(parallel=True)
def solve(t,ma,st,ra,d,sol_c,sol_m,sol_v,par):
    """ solve the model in the last period for singles """
        
    # unpack (helps numba optimize)
    c = sol_c[t,ra,d,:]
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
        v[i] = utility.func(c[i],t,d,ma,st,par) + par.beta*par.gamma*(m[i]-c[i])


@njit(parallel=True)
def solve_c(t,ad,st_h,st_w,ra_h,ra_w,d_h,d_w,sol_c,sol_m,sol_v,par):
    """ solve the model in the last period for couples """
        
    # unpack (helps numba optimize)
    ad_idx = ad+par.ad_min
    d = transitions.d_c(d_h,d_w)
    c = sol_c[t,ad_idx,st_h,st_w,ra_h,ra_w,d,:]
    m = sol_m[:]
    v = sol_v[t,ad_idx,st_h,st_w,ra_h,ra_w,d,:]

    # initialize
    c[:] = m[:]
    
    # optimal choice
    cons = (par.beta*par.gamma)**(-1/par.rho)
    for i in range(len(m)):
        if m[i] > cons:
            c[i] = cons
        else:
            c[i] = m[i]
        v[i] = utility.func_c(c[i],t,ad,d_h,d_w,st_h,st_w,par) + par.beta*par.gamma*(m[i]-c[i])

    

