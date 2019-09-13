# global modules
from numba import njit

# local modules
import transitions

@njit(parallel=True)
def func(c,d,st,par):
    """ utility function 
    
        Args:

            c=consumption, d=retirement choice
            st=state, par=parameters"""    

    if d == 0:  # retired
        hs = transitions.state_translate(st,'high_skilled',par)
        if transitions.state_translate(st,'male',par) == 1:
            leisure = par.alpha_0_male + hs*par.alpha_1
        else:
            leisure = par.alpha_0_female + hs*par.alpha_1
        
    else:       # working
        leisure = 0
              
    return c**(1-par.rho)/(1-par.rho) + leisure


@njit(parallel=True)
def marg_func(c,par):     
    return c**(-par.rho)

@njit(parallel=True)
def inv_marg_func(u,par):
    return u**(-1/par.rho)