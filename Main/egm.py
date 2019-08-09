from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope 

# local modules
import utility

@njit(parallel=True)
def solve_bellman_retired(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    c = sol.c[t]
    m = sol.m[t]
    v = sol.v[t]
    q = sol.q[t]
    v_plus_raw = sol.v_plus_raw[t]

    # a. solution
    c[:,0] = utility.inv_marg_func(q[:,0],par)
    m[:,0] = par.grid_a + c[:,0]
    v[:,0] = utility.func(c[:,0],0,par) + par.beta*v_plus_raw[:,0]

@njit(parallel=True)
def solve_bellman_work(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    c = sol.c[t]
    m = sol.m[t]
    v = sol.v[t]
    q = sol.q[t]
    v_plus_raw = sol.v_plus_raw[t]

    # a. raw solution
    for id in prange(2): # in parallel
        if id == 0: # retired
            c[:,id] = utility.inv_marg_func(q[:,id],par)
            m[:,id] = par.grid_a + c[:,id]
            v[:,id] = utility.func(c[:,id],id,par) + par.beta*v_plus_raw[:,id]
        else:
            c_raw = utility.inv_marg_func(q[:,id],par)
            m_raw = par.grid_a + c_raw
            v_raw = utility.func(c_raw,id,par) + par.beta*v_plus_raw[:,id]

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
