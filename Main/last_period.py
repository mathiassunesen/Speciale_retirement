from numba import njit, prange
import numpy as np

# local modules
import utility

@njit(parallel=True)
def solve(t,st,sol,par):
    """ solve the problem in the last period """

    # unpack (helps numba optimize)
    m = sol.m[t,st,:,0]
    v = sol.v[t,st,:,0]
    c = sol.c[t,st,:,0]

    # initialize
    m[:] = np.linspace(1e-6,par.a_max,par.Na+par.poc)
    c[:] = m[:]
    
    # optimal choice
    cons = (par.beta*par.gamma)**(-1/par.rho)
    for i in prange(len(m)):
        if m[i] > cons:
                c[i] = cons
        else:
                c[i] = m[i]
    
    #for i in reversed(range(len(m))):
    #    if m[i] > cons:
    #            c[i] = cons
    #    else:
    #            break
    
    # optimal value
    v[:] = utility.func(c,0,st,par) + par.beta*par.gamma*(m-c)



    

