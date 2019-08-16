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
         if a[i+1] < a[i]:
               return False
    return True

@njit(parallel=True)
def solve_bellman_retired(t,st,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,st,poc:,0] # leave/ignore points on constraint
    m = sol.m[t,st,poc:,0]
    v = sol.v[t,st,poc:,0]
    q = sol.q[t,st,:,0]
    v_plus_raw = sol.v_plus_raw[t,st,:,0]
    pi = transitions.survival(t,par)     

    # a. solution
    c[:] = utility.inv_marg_func(q,par)
    m[:] = par.grid_a + c
    v[:] = utility.func(c,0,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*par.grid_a)   

    # b. add points on constraint
    points_on_constraint(t,st,0,sol,par)    

@njit(parallel=True)
def solve_bellman_work(t,st,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,st,poc:] # ignore/leave points on constraint
    m = sol.m[t,st,poc:]
    v = sol.v[t,st,poc:]
    q = sol.q[t,st]
    v_plus_raw = sol.v_plus_raw[t,st]
    pi = transitions.survival(t,par)     

    # a. raw solution
    for id in prange(2): # in parallel
        if id == 0: # retired
            c[:,id] = utility.inv_marg_func(q[:,id],par)
            m[:,id] = par.grid_a + c[:,id]
            v[:,id] = utility.func(c[:,id],id,st,par) + par.beta*(pi*v_plus_raw[:,id] + (1-pi)*par.gamma*par.grid_a)
        else:
            c_raw = utility.inv_marg_func(q[:,id],par)
            m_raw = par.grid_a + c_raw
            v_raw = utility.func(c_raw,id,st,par) + par.beta*(pi*v_plus_raw[:,id] + (1-pi)*par.gamma*par.grid_a)

    # b. re-interpolate to common grid
    idx = np.argsort(m_raw)
    m[:,1] = m_raw[idx]

    if is_sorted(idx): # no need for upper envelope
        c[:,1] = c_raw
        v[:,1] = v_raw
    else:
        print('envelope')
        envelope = upperenvelope.create(utility.func)
        envelope(par.grid_a,m_raw,c_raw,v_raw,m[:,1], # input
                 c[:,1],v[:,1], # output
                 1,par) # args for utility function

    # c. add points on constraint
    points_on_constraint(t,st,0,sol,par)    
    points_on_constraint(t,st,1,sol,par)

    
@njit(parallel=True)
def points_on_constraint(t,st,d,sol,par):
    """add points on the constraint"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    low_c = sol.c[t,st,poc,d] # lowest point of the inner solution
    c = sol.c[t,st,:poc,d] # only consider points on constraint
    m = sol.m[t,st,:poc,d]
    v = sol.v[t,st,:poc,d]

    # add points on constraint
    if low_c > 1e-6:
        c[:] = np.linspace(1e-6,low_c-1e-6,poc)
    else:
        c[:] = np.linspace(low_c/3,low_c/2,poc)
    m[:] = c[:]
    
    if d == 0:
        v[:] = post_decision.value_of_choice_retired(t,st,m,c,sol,par)
    else:
        v[:] = post_decision.value_of_choice_work(t,st,m,c,sol,par)

