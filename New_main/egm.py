from numba import njit, prange
import numpy as np

# consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope 

# local modules
import utility
import post_decision
import transitions

@njit(parallel=True)
def is_sorted(a): # fast implementation
    for i in range(a.size-1):
         if a[i+1] < a[i] :
               return False
    return True

@njit(parallel=True)
def solve_bellman_retired(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,poc:,0] # leave/ignore points on constraint
    m = sol.m[t,poc:,0]
    v = sol.v[t,poc:,0]
    q = sol.q[t,:,0]
    v_plus_raw = sol.v_plus_raw[t,:,0]

    # a. solution
    c[:] = utility.inv_marg_func(q,par)
    m[:] = par.grid_a + c
    v[:] = utility.func(c,0,par) + par.beta*v_plus_raw

@njit(parallel=True)
def solve_bellman_work(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,poc:] # ignore/leave points on constraint
    m = sol.m[t,poc:]
    v = sol.v[t,poc:]
    q = sol.q[t]
    v_plus_raw = sol.v_plus_raw[t]
    pi = transitions.survival(t,par)     

    # a. raw solution
    for id in prange(2): # in parallel
        if id == 0: # retired
            c[:,id] = utility.inv_marg_func(q[:,id],par)
            m[:,id] = par.grid_a + c[:,id]
            v[:,id] = utility.func(c[:,id],id,par) + par.beta*(pi*v_plus_raw[:,id] + (1-pi)*par.gamma*par.grid_a)
        else:
            c_raw = utility.inv_marg_func(q[:,id],par)
            m_raw = par.grid_a + c_raw
            v_raw = utility.func(c_raw,id,par) + par.beta*(pi*v_plus_raw[:,id] + (1-pi)*par.gamma*par.grid_a)

    # b. re-interpolate to common grid
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

    # c. add points on constraint
    points_on_constraint(t,sol,par)

    
@njit(parallel=True)
def points_on_constraint(t,sol,par):
    """add points on the constraint"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    low_c = sol.c[t,poc,1] # lowest point of the inner solution
    c = sol.c[t,:poc,1] # only consider points on constraint
    m = sol.m[t,:poc,1]
    v = sol.v[t,:poc,1]

    # add points on constraint
    c[:] = np.linspace(1e-6,low_c-1e-6,poc)
    m[:] = c[:]
    v[:] = post_decision.value_of_choice(t,m,c,sol,par)

