# global modules
from numba import njit, prange
import numpy as np

# local modules
import utility

@njit(parallel=True)
def solve(t,st,ad,sol,par):
    """ solve the problem in the last period 

    Args:

        t=time, st=states, ad=age difference, sol=solution, par=parameters
        """
        
    # unpack (helps numba optimize)
    c = sol.c[t,st,ad,:,0]
    m = sol.m[t,st,ad,:,0]
    v = sol.v[t,st,ad,:,0]

    # initialize
    m[:] = np.linspace(par.tol,par.a_max,par.Na+par.poc)
    c[:] = m[:]
    
    # optimal choice
    cons = (par.beta*par.gamma)**(-1/par.rho)
    for i in range(len(m)):
        if m[i] > cons:
            c[i] = cons
        else:
            c[i] = m[i]
        v[i] = utility.func(c[i],0,st,par) + par.beta*par.gamma*(m[i]-c[i])


    

