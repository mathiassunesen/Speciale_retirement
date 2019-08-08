import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope 

# local modules
import utility

@njit(parallel=True)
def is_sorted(a): # fast implementation
    for i in range(a.size-1):
         if a[i+1] < a[i] :
               return False
    return True

@njit(parallel=True)
def solve_bellman(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    c = sol.c[t]
    m = sol.m[t]
    v = sol.v[t]
    q = sol.q[t]
    v_plus_raw = sol.v_plus_raw[t]

    # a. raw solution
    for iz in prange(2): # in parallel
        if iz == 0: # retired
            c[:,iz] = utility.inv_marg_func(q[:,iz],par)
            m[:,iz] = par.grid_a + c[:,iz]
            v[:,iz] = utility.func(c[:,iz],iz,par) + par.beta*v_plus_raw[:,iz]
        else:
            c_raw = utility.inv_marg_func(q[:,iz],par)
            m_raw = par.grid_a + c_raw
            v_raw = utility.func(c_raw,iz,par) + par.beta*v_plus_raw[:,iz]

    # b. re-interpolate to common grid (only for working)
    idx = np.argsort(m_raw)
    m[:,1] = m_raw[idx]

    if is_sorted(idx): # no need for upper envelope
        c[:,1] = c_raw
        v[:,1] = v_raw
    else:
        print('envelope')
        print('')
        envelope = upperenvelope.create(utility.func)
        envelope(par.grid_a,m_raw,c_raw,v_raw,m[:,1], # input
                 c[:,1],v[:,1], # output
                 1,par) # args for utility function