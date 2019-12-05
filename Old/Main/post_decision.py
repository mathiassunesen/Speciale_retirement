# global modules
import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp

# local modules
import utility
import funs
import transitions
import egm


###############################
### Functions for singles #####
###############################
@njit(parallel=True)
def compute(t,ma,st,ra,D,sol_c,sol_m,sol_v,a,par):
    """ compute post decision states v_plus_raw and q, which is used to solve the bellman equation for singles"""        

    # unpack solution
    c = sol_c[t+1]
    m = sol_m[:]
    v = sol_v[t+1]

    # prep
    v_plus_raw = np.nan*np.zeros((2,len(a)))
    avg_marg_u_plus = np.nan*np.zeros((2,len(a)))
    Ra = par.R*a    
       
    # loop over the choices
    for d in D:  

        # next period choice set and retirement age
        d_plus = transitions.d_plus(t,d,par)
        ra_plus = transitions.ra_look_up(t+1,st,ra,d,par)

        # next period income
        inc = transitions.inc_lookup_single(d,t+1,ma,st,ra,par)    

        # weights
        if d == 0:
            w = np.array([1.0])     # no integration
        elif d == 1:
            w = par.xi_w[ma]

        # c. integration            
        v_plus_raw[d],avg_marg_u_plus[d] = shocks_GH(t,Ra,inc,w,c[ra_plus],m[:],v[ra_plus],par,d_plus)
    
    # return
    return v_plus_raw,avg_marg_u_plus


###############################
### Functions for couples #####
###############################
@njit(parallel=True)
def compute_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
              sol_c,sol_m,sol_v,
              single_sol_v_plus_raw,single_sol_avg_marg_u_plus):
    """ compute post decision for couples""" 

    # unpack solution
    ad_min = par.ad_min
    ad_idx = ad+ad_min
    c = sol_c[t+1,ad_idx,st_h,st_w]
    m = sol_m[:]
    v = sol_v[t+1,ad_idx,st_h,st_w]  

    # unpack single solution
    v_plus_raw_h = single_sol_v_plus_raw[t+1+ad_min,1,st_h,ra_h]                  #  ma=1
    avg_marg_u_plus_h = single_sol_avg_marg_u_plus[t+1+ad_min,1,st_h,ra_h]        #  ma=1   
    v_plus_raw_w = np.zeros(v_plus_raw_h.shape)                                   # initialize            
    avg_marg_u_plus_w = np.zeros(avg_marg_u_plus_h.shape)                         # initialize   
    if t+1+ad < par.T:    # wife alive   
        v_plus_raw_w = single_sol_v_plus_raw[t+1+ad_idx,0,st_w,ra_w]              # ma=0
        avg_marg_u_plus_w = single_sol_avg_marg_u_plus[t+1+ad_idx,0,st_w,ra_w]    # ma=0   

    # prep
    v_raw = np.nan*np.zeros((4,len(a)))
    q = np.nan*np.zeros((4,len(a)))
    Ra = par.R*a    
    pi_plus_h,pi_plus_w = transitions.survival_lookup_couple(t+1,ad,st_h,st_w,par)        

    # loop over the choices
    for d_h in D_h:
        for d_w in D_w:

            # next period choice set and retirement age
            d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
            ra_plus_h = transitions.ra_look_up(t+1,st_h,ra_h,d_h,par)       # husband            
            ra_plus_w = transitions.ra_look_up(t+1+ad,st_w,ra_w,d_w,par)    # wife

            # next period income
            inc = transitions.inc_lookup_couple(d_h,d_w,t+1,ad,st_h,st_w,ra_h,ra_w,par)  

            # weights
            if d_h == 0 and d_w == 0:
                w = np.array([1.0])     # no integration
            elif d_h == 0 and d_w == 0:
                w = par.xi_w[0]         # wife
            elif d_h == 1 and d_w == 0:
                w = par.xi_w[1]         # husband
            elif d_h == 1 and d_w == 1:
                w = par.w_corr          # joint

            # interpolate/integrate   
            v_plus_raw_c,avg_marg_u_plus_c = shocks_GH(t,Ra,inc,w,c[ra_plus_h,ra_plus_w],m[:],v[ra_plus_h,ra_plus_w],par,d_plus)
    
            # indices to look up
            d = transitions.d_c(d_h,d_w)                    # joint index     
            d_plus_h = transitions.d_plus_int(t,d_h,par)    # single, husband
            d_plus_w = transitions.d_plus_int(t+ad,d_w,par)    # single, wife
            v_raw[d] = par.beta*(pi_plus_h*pi_plus_w*v_plus_raw_c +
                                (1-pi_plus_w)*pi_plus_h*v_plus_raw_h[d_plus_h] + 
                                (1-pi_plus_h)*pi_plus_w*v_plus_raw_w[d_plus_w] + 
                                (1-pi_plus_h)*(1-pi_plus_w)*par.gamma*a)
            
            q[d] = par.beta*(par.R*(pi_plus_h*pi_plus_w*avg_marg_u_plus_c +
                                   (1-pi_plus_w)*pi_plus_h*avg_marg_u_plus_h[d_plus_h] + 
                                   (1-pi_plus_h)*pi_plus_w*avg_marg_u_plus_w[d_plus_w]) + 
                                   (1-pi_plus_w)*(1-pi_plus_h)*par.gamma)  

    # return
    return v_raw,q           


###############################
###       Integration     #####
###############################
@njit(parallel=True)
def shocks_GH(t,inc_no_shock,inc,w,c,m,v,par,d_plus):     
    """ compute v_plus_raw and avg_marg_u_plus using GaussHermite integration if necessary """    
    
    # a. initialize  
    c_plus_interp = np.zeros((4,inc_no_shock.size))
    v_plus_interp = np.zeros((4,inc_no_shock.size)) 
    v_plus_raw = np.zeros(inc_no_shock.size)
    avg_marg_u_plus = np.zeros(inc_no_shock.size)
    prep = linear_interp.interp_1d_prep(len(inc_no_shock))    # save the position of numbers to speed up interpolation

    # b. loop over GH-nodes
    for i in range(len(w)):
        m_plus = inc_no_shock + inc[i]

        # 1. interpolate
        for d in d_plus:
            linear_interp.interp_1d_vec_mon(prep,m[:],c[d,:],m_plus,c_plus_interp[d,:])
            linear_interp.interp_1d_vec_mon_rep(prep,m[:],v[d,:],m_plus,v_plus_interp[d,:])         

        # 2. logsum and v_plus_raw
        if len(d_plus) == 1:     # no taste shocks
            v_plus_raw += w[i]*v_plus_interp[d_plus[0],:]
            avg_marg_u_plus += w[i]*utility.marg_func(c_plus_interp[d_plus[0],:],par)

        elif len(d_plus) == 2:   # taste shocks
            logsum,prob = funs.logsum2(v_plus_interp[d_plus,:],par)           
            v_plus_raw += w[i]*logsum[0,:]
            marg_u_plus = prob[d_plus[0],:]*utility.marg_func(c_plus_interp[d_plus[0],:],par) + (1-prob[d_plus[0],:])*utility.marg_func(c_plus_interp[d_plus[1],:],par)
            avg_marg_u_plus += w[i]*marg_u_plus    

        elif len(d_plus) == 4:   # both are working
            logsum,prob = funs.logsum4(v_plus_interp[d_plus,:],par)
            v_plus_raw += w[i]*logsum[0,:]
            marg_u_plus = (prob[0,:]*utility.marg_func(c_plus_interp[0,:],par) + 
                           prob[1,:]*utility.marg_func(c_plus_interp[1,:],par) + 
                           prob[2,:]*utility.marg_func(c_plus_interp[2,:],par) +
                           prob[3,:]*utility.marg_func(c_plus_interp[3,:],par))  
            avg_marg_u_plus += w[i]*marg_u_plus        

    # c. return results
    return v_plus_raw,avg_marg_u_plus          