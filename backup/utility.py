from numba import njit
import transitions

@njit
def func(c,d,st,par):

    if par.couples:
        
        if d == 0: # both retired
            hs_f = transitions.state_translate(st,'female_high_skilled',par)
            hs_m = transitions.state_translate(st,'male_high_skilled',par)                        
            leisure_f = (par.alpha_0_female + hs_f*par.alpha_1)*(1 + par.phi_0_female + hs_f*par.phi_1)
            leisure_m = (par.alpha_0_male + hs_m*par.alpha_1)*(1 + par.phi_0_male + hs_m*par.phi_1)
                        
        elif d == 1: # wife working, husband retired
            hs_m = transitions.state_translate(st,'male_high_skilled',par)
            leisure_m = par.alpha_0_male + hs_m*par.alpha_1
            leisure_f = 0

        elif d == 2: # wife retired, husband working
            hs_f = transitions.state_translate(st,'male_high_skilled',par)
            leisure_f = par.alpha_0_female + hs_f*par.alpha_1
            leisure_m = 0

        elif d == 3: # both working
            leisure_f = 0
            leisure_m = 0

        cons = c**(1-par.rho)/(1-par.rho)
        return par.pareto_w*(cons + leisure_m) + (1 - par.pareto_w)*(cons + leisure_f)

    else: # singles
        
        if d == 0: # retired
            hs = transitions.state_translate(st,'high_skilled',par)
            ch = transitions.state_translate(st,'children',par)
            if transitions.state_translate(st,'male',par) == 1:
                leisure = par.alpha_0_male + hs*par.alpha_1 + ch*par.alpha_2
            else:
                leisure = par.alpha_0_female + hs*par.alpha_1 + ch*par.alpha_2
        
        else: # working
            leisure = 0
              
        return c**(1-par.rho)/(1-par.rho) + leisure

@njit
def marg_func(c,par): # assuming pareto_w=0.5
    return c**(-par.rho)

@njit
def inv_marg_func(u,par): # assuming pareto_w=0.5
    return u**(-1/par.rho)