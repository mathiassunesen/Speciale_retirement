from numba import njit, prange
import numpy as np

# local modules
import utility

@njit(parallel=True)
def solve(t,sol,par):
    """ solve the problem in the last period """

    # unpack (helps numba optimize)
    m = sol.m[t,:,0]
    v = sol.v[t,:,0]
    c = sol.c[t,:,0]

    # last period
    m = np.linspace(1e-6,par.a_max,par.Na)
    c = np.linspace(1e-6,par.a_max,par.Na)
    cons = (par.beta*par.gamma)**(-1/par.rho)
    for i in reversed(range(len(m))):
        if m[i] > cons:
                c[i] = cons
        else:
                break

    v = par.gamma*(m-c)



    

