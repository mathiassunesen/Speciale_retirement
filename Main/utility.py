from numba import njit

@njit
def func(c,d,par):
    return c**(1-par.rho)/(1-par.rho) + (d == 0) * par.alpha

@njit
def marg_func(c,par):
    return c**(-par.rho)

@njit
def inv_marg_func(u,par):
    return u**(-1/par.rho)