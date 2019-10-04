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

# upper envelope function
envelope = upperenvelope.create(utility.func)
envelope_c = upperenvelope.create(utility.func_c)

###############################
### Functions for singles #####
###############################
@njit(parallel=True)
def solve(t,ad,ma,st,ra,d,sol,par):
    """ wrapper which calls both post_decision.compute and egm.solve_bellman"""

    post_decision.compute(t,ad,ma,st,ra,d,sol,par)
    solve_bellman(t,ad,ma,st,ra,d,sol,par)  

@njit(parallel=True)
def solve_bellman(t,ad,ma,st,ra,d_lst,sol,par):
    """ solve the bellman equation for singles using the endogenous grid method"""

    # unpack solution
    c = sol.c[t,ad,ma,st,ra]
    m = sol.m[t,ad,ma,st,ra]
    v = sol.v[t,ad,ma,st,ra]

    # unpack rest
    a = par.grid_a
    q = sol.q[:,:]
    v_plus_raw = sol.v_plus_raw[:,:]        
    pi = transitions.survival_look_up(t,ma,par)   

    # loop over the choices
    for d in d_lst:

        # a. raw solution
        c_raw = utility.inv_marg_func(q[d,:],par)
        m_raw = a + c_raw
        v_raw = par.beta*(pi*v_plus_raw[d,:] + (1-pi)*par.gamma*a)

        # b. upper envelope
        m[d,:] = a      
        envelope(a,m_raw,c_raw,v_raw,m[d,:],    # input
                 c[d,:],v[d,:],                 # output
                 d,ma,st,par)                   # args for utility function  


@njit(parallel=True)
def solve_c(t,ad,st_h,st_w,d,sol,par,single_sol,retirement=[0,0,0]):
    """ wrapper which calls both post_decision.compute_c and egm.solve_bellman_c"""
        
    post_decision.compute_c(t,ad,st_h,st_w,d,sol,par,retirement)
    solve_bellman_c(t,ad,st_h,st_w,d,sol,par,single_sol,retirement)  


###############################
### Functions for couples #####
###############################
@njit(parallel=True)
def solve_bellman_c(t,ad,st_h,st_w,d,sol,par,single_sol,retirement):
    """ solve the bellman equation for couples using the endogenous grid method"""

    # unpack solution
    ad_min = par.ad_min
    ad_idx = ad+ad_min
    c = sol.c[t,ad_idx,st_h,st_w]
    m = sol.m[t,ad_idx,st_h,st_w]
    v = sol.v[t,ad_idx,st_h,st_w]

    # unpack rest
    a = par.grid_a
    q = sol.q[:,:]
    v_plus_raw = sol.v_plus_raw[:,:]        
    pi_w,pi_h = transitions.survival_look_up_c(t,ad,par)   

    # loop over the choices
    for idx in d:

        # retirement age
        # if id == 0:
        #     erp = retirement[2]
        # elif id == 1:
        #     erp = 0
        erp = retirement[2]            

        # looking up in single solution
        sid_h,sid_w = transitions.couple_index(idx,t,ad,par)    # for look up in single solution  
        VH = single_sol.v[t+1+ad_min,0,1,st_h,erp,sid_h]        # ad=0 and male=1
        VW = np.zeros(VH.shape)     # initialize
        if t+1+ad < par.T:          # wife alive
            VW[:] = single_sol.v[t+1+ad_idx,0,0,st_w,erp,sid_w] # ad=0 and male=0     

        # a. raw solution
        c_raw = utility.inv_marg_func(q[idx],par)
        m_raw = a + c_raw
        v_raw = par.beta*(pi_h*pi_w*v_plus_raw[idx] +       # both alive
                          pi_h*(1-pi_w)*VH +             # husband alive -> look up in single solution
                         (1-pi_h)*pi_w*VW +              # wife alive -> look up in single solution
                         (1-pi_h)*(1-pi_w)*par.gamma*a)  # both dead

        # b. upper envelope
        m[erp,idx,:] = a      
        envelope_c(a,m_raw,c_raw,v_raw,m[erp,idx],   # input
                   c[erp,idx],v[erp,idx],            # output
                   idx,st_h,st_w,par)                   # args for utility function  