# global modules
from numba import njit, prange
import numpy as np

# consav
from consav import linear_interp
from consav import upperenvelope 

# local modules
import utility
import post_decision
import transitions
import funs

# upper envelope function
envelope = upperenvelope.create(utility.func)
envelope_c = upperenvelope.create(utility.func_c)

###############################
### Functions for singles #####
###############################
@njit(parallel=True)
def solve_bellman(t,ma,st,ra,D,sol_c,sol_m,sol_v,sol_v_plus_raw,sol_avg_marg_u_plus,a,par):
    """ solve the bellman equation for singles"""    

    # compute post decision (and store results, since they are needed in couple model)
    v_plus_raw,avg_marg_u_plus = post_decision.compute(t,ma,st,ra,D,sol_c,sol_m,sol_v,a,par)    
    sol_v_plus_raw[t+1,ra] = v_plus_raw
    sol_avg_marg_u_plus[t+1,ra] = avg_marg_u_plus    
    
    # unpack
    c = sol_c[t,ra]
    m = sol_m[:]
    v = sol_v[t,ra]        
    pi_plus = transitions.survival_lookup_single(t+1,ma,st,par)   

    # loop over the choices
    for d in D:

        # a. post decision
        q = par.beta*(par.R*pi_plus*avg_marg_u_plus[d] + (1-pi_plus)*par.gamma)

        # b. raw solution
        c_raw = utility.inv_marg_func(q,par)
        m_raw = a + c_raw
        v_raw = par.beta*(pi_plus*v_plus_raw[d] + (1-pi_plus)*par.gamma*a)  # without utility (added in envelope)

        # c. upper envelope
        envelope(a,m_raw,c_raw,v_raw,m,     # input
                 c[d],v[d],                 # output
                 d,ma,st,par)               # args for utility function  


###############################
### Functions for couples #####
###############################
@njit(parallel=True)
def solve_bellman_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
                    sol_c,sol_m,sol_v,
                    single_sol_v_plus_raw,single_sol_avg_marg_u_plus):
    """ solve the bellman equation for singles"""    

    # compute post decision
    v_raw,q = post_decision.compute_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
                                      sol_c,sol_m,sol_v,
                                      single_sol_v_plus_raw,single_sol_avg_marg_u_plus)

    # unpack solution
    ad_min = par.ad_min
    ad_idx = ad+ad_min
    c = sol_c[t,ad_idx,st_h,st_w,ra_h,ra_w]
    m = sol_m[:]
    v = sol_v[t,ad_idx,st_h,st_w,ra_h,ra_w]

    # loop over the choices
    for d_h in D_h:
        for d_w in D_w:

            # a. indices
            d = transitions.d_c(d_h,d_w)                # joint index

            # b. raw solution
            c_raw = utility.inv_marg_func(q[d],par)
            m_raw = a + c_raw

            # d. upper envelope
            envelope_c(a,m_raw,c_raw,v_raw[d],m,        # input
                       c[d],v[d],                       # output
                       d_h,d_w,st_h,st_w,par)           # args for utility function  