from numba import njit, prange

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