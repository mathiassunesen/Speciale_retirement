from numba import njit
import transitions

@njit
def func(c,d,st,par):

    if (d == 1): # working
        leisure = 0
    else: # retired
        ma = transitions.state_translate(st,'male',par)
        hs = transitions.state_translate(st,'high_skilled',par)
        ch = transitions.state_translate(st,'children',par)                
        leisure = (ma == 1)*par.alpha_0_male + (ma == 0)*par.alpha_0_female + hs*par.alpha_1 + ch*par.alpha_2 

    return c**(1-par.rho)/(1-par.rho) + leisure

@njit
def marg_func(c,par):
    return c**(-par.rho)

@njit
def inv_marg_func(u,par):
    return u**(-1/par.rho)