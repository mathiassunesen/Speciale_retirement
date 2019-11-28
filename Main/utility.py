# global modules
from numba import njit

# local modules
import transitions

@njit(parallel=True)
def func(c,d,ma,st,par):
    """ utility function for singles"""    

    if d == 0:  # retired
        hs = transitions.state_translate(st,'high_skilled',par)
        if ma == 1:
            leisure = par.alpha_0_male + hs*par.alpha_1
        elif ma == 0:
            leisure = par.alpha_0_female + hs*par.alpha_1
        
    else:       # working
        leisure = 0
              
    return c**(1-par.rho)/(1-par.rho) + leisure


@njit(parallel=True)
def func_c(c,d_h,d_w,st_h,st_w,par):
    """ utility function for couples"""    
    
    # high skilled
    hs_h = transitions.state_translate(st_h,'high_skilled',par)
    hs_w = transitions.state_translate(st_w,'high_skilled',par)
    
    # initialize to both working (then overwrite if not)
    alpha_h = 0.0
    phi_h = 0.0
    alpha_w = 0.0
    phi_w = 0.0

    if d_h == 0 and d_w == 0:      # both retired
        alpha_h = par.alpha_0_male + hs_h*par.alpha_1
        phi_h = par.phi_0_male + hs_h*par.phi_1
        alpha_w = par.alpha_0_female + hs_w*par.alpha_1
        phi_w = par.phi_0_female + hs_w*par.phi_1

    elif d_h == 0 and d_w == 1:    # only husband retired
        alpha_h = par.alpha_0_male + hs_h*par.alpha_1

    elif d_h == 1 and d_w == 0:    # only wife retired
        alpha_w = par.alpha_0_female + hs_w*par.alpha_1

    lei_h = alpha_h*(1 + phi_h)
    lei_w = alpha_w*(1 + phi_w)
    n = 1 + par.v # equivalence scale
    Crho = (c/n)**(1-par.rho)/(1-par.rho)
    w = par.pareto_w
    return w*(Crho + lei_h) + (1-w)*(Crho + lei_w)

@njit(parallel=True)
def marg_func(c,par):     
    # return c**(-par.rho)
    n = 1 + par.v*par.couple
    return n*(c/n)**(-par.rho)

@njit(parallel=True)
def inv_marg_func(u,par):
    # return u**(-1/par.rho)
    n = 1 + par.v*par.couple
    return n*(u/n)**(-1/par.rho)