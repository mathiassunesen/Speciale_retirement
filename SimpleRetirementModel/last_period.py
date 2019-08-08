from numba import njit, prange

# local modules
import utility

@njit(parallel=True)
def solve(t,sol,par):
    """ solve the problem in the last period """

    # unpack (helps numba optimize)
    m = sol.m[t]
    v = sol.v[t]
    c = sol.c[t]

    # loop over states
    for iz in prange(2): # in parallel
                    
        # a. states    
        m[:,iz] = par.grid_a

        # b. optimal choice (consume everything)
        c[:,iz] = par.grid_a

        # c. optimal value
        v[:,iz] = utility.func(c[:,iz],iz,par)
