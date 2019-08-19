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
def solve_bellman_retired(t,st,sol,par,retirement):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,st,poc:,0,retirement[2]] # leave/ignore points on constraint
    m = sol.m[t,st,poc:,0,retirement[2]]
    v = sol.v[t,st,poc:,0,retirement[2]]
    q = sol.q[t,st,:,0,retirement[2]]
    v_plus_raw = sol.v_plus_raw[t,st,:,0,retirement[2]]
    pi = transitions.survival(t,par)     

    # a. solution
    c[:] = utility.inv_marg_func(q,par)
    m[:] = par.grid_a + c
    v[:] = utility.func(c,0,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*par.grid_a)   

    # b. add points on constraint
    points_on_constraint(t,st,0,sol,par,retirement)    

@njit(parallel=True)
def solve_bellman_work(t,st,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,st,poc:,1,0] # ignore/leave points on constraint
    m = sol.m[t,st,poc:,1,0]
    v = sol.v[t,st,poc:,1,0]
    q = sol.q[t,st,:,1,0]
    v_plus_raw = sol.v_plus_raw[t,st,:,1,0]
    pi = transitions.survival(t,par)     

    # a. raw solution
    c_raw = utility.inv_marg_func(q,par)
    m_raw = par.grid_a + c_raw
    v_raw = utility.func(c_raw,1,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*par.grid_a)

    # b. re-interpolate to common grid
    idx = np.argsort(m_raw)
    m[:] = m_raw[idx]

    if is_sorted(idx): # no need for upper envelope
        c[:] = c_raw
        v[:] = v_raw
    else:
        print('envelope')
        envelope = upperenvelope.create(utility.func)
        envelope(par.grid_a,m_raw,c_raw,v_raw,m, # input
                 c,v, # output
                 1,par) # args for utility function

    # c. add points on constraint 
    points_on_constraint(t,st,1,sol,par,[0,0,0])
    
@njit(parallel=True)
def points_on_constraint(t,st,d,sol,par,retirement):
    """add points on the constraint"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    low_c = sol.c[t,st,poc,d,retirement[2]] # lowest point of the inner solution
    c = sol.c[t,st,:poc,d,retirement[2]] # only consider points on constraint
    m = sol.m[t,st,:poc,d,retirement[2]]
    v = sol.v[t,st,:poc,d,retirement[2]]

    # add points on constraint
    if low_c > 1e-6:
        c[:] = np.linspace(1e-6,low_c-1e-6,poc)
    else:
        c[:] = np.linspace(low_c/3,low_c/2,poc)
    m[:] = c[:]
    
    if d == 0:
        v[:] = post_decision.value_of_choice_retired(t,st,m,c,sol,par,retirement)
    else:
        v[:] = post_decision.value_of_choice_work(t,st,m,c,sol,par)


@njit(parallel=True)
def all_egm(t,st,sol,par,retirement):
    """run all functions"""

    post_decision.compute_retired(t,st,sol,par,retirement)
    solve_bellman_retired(t,st,sol,par,retirement)  
    post_decision.compute_work(t,st,sol,par)                    
    solve_bellman_work(t,st,sol,par)   