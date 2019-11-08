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
def compute(t,ad,ma,st,ra,D,sol_c,sol_m,sol_v,a,par):
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

        # a. retire
        if d == 0:

            # next period choice set and retirement age
            d_plus = transitions.d_plus(t,d,par)
            ra_plus = transitions.ra_look_up(t+1,st,ra,d,par)
            
            # next period income
            w = np.array([1.0]) # no integration
            inc = transitions.inc_lookup_single(d,t+1,ma,st,ra,par)                           
        
        # b. work
        elif d == 1:
            
            # next period choice set and retirement age 
            d_plus = transitions.d_plus(t,d,par)
            ra_plus = transitions.ra_look_up(t+1,st,ra,d,par)

            # next period income
            w = par.xi_w[ma]
            inc = transitions.inc_lookup_single(d,t+1,ma,st,ra,par)  

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
              single_sol_v_plus_raw,single_sol_avg_marg_u_plus,look_up=True):
    """ compute post decision for couples""" 

    # unpack solution
    ad_min = par.ad_min
    ad_idx = ad+ad_min
    c = sol_c[t+1,ad_idx,st_h,st_w]
    m = sol_m[t+1,ad_idx,st_h,st_w]
    v = sol_v[t+1,ad_idx,st_h,st_w]  

    # unpack single solution
    v_plus_raw_h = single_sol_v_plus_raw[t+1+ad_min,0,1,st_h,ra_h]                  # ad=0 and male=1
    avg_marg_u_plus_h = single_sol_avg_marg_u_plus[t+1+ad_min,0,1,st_h,ra_h]        # ad=0 and male=1   
    v_plus_raw_w = np.zeros(v_plus_raw_h.shape)                                     # initialize            
    avg_marg_u_plus_w = np.zeros(avg_marg_u_plus_h.shape)                           # initialize   
    if t+1+ad < par.T:    # wife alive   
        v_plus_raw_w = single_sol_v_plus_raw[t+1+ad_idx,0,0,st_w,ra_w]              # ad=0 and male=1
        avg_marg_u_plus_w = single_sol_avg_marg_u_plus[t+1+ad_idx,0,0,st_w,ra_w]    # ad=0 and male=1   

    # prep
    v_raw = np.nan*np.zeros((4,len(a)))
    q = np.nan*np.zeros((4,len(a)))
    Ra = par.R*a    
    pi_plus_h,pi_plus_w = transitions.survival_look_up_c(t+1,ad,par)        

    # loop over the choices
    for d_h in D_h:
        for d_w in D_w:

            # a. both retire
            if d_h == 0 and d_w == 0:
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_plus_h = transitions.ra_look_up(t+1,st_h,ra_h,d_h,par)       # husband            
                ra_plus_w = transitions.ra_look_up(t+1+ad,st_w,ra_w,d_w,par)    # wife

                # income
                if look_up:
                    pens_h = transitions.pension_look_up_c(d_h,d_w,t+1,ad,1,st_h,st_w,ra_h,ra_w,par)  # husband
                    pens_w = transitions.pension_look_up_c(d_h,d_w,t+1,ad,0,st_h,st_w,ra_h,ra_w,par)  # wife
                else:
                    pass
                    # pens_h = transitions.pension(t+1,1,st_h,ra_h,a,par)             # husband
                    # pens_w = transitions.pension(t+1+ad,0,st_w,ra_w,a,par)          # wife

                inc_no_shock = Ra[:] + pens_h + pens_w
                w = np.array([1.0]) # no integration
                inc = 0*w[:]

            # b. husband retire, wife working
            elif d_h == 0 and d_w == 1:                  
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_plus_h = transitions.ra_look_up(t+1,st_h,ra_h,d_h,par)       # husband            
                ra_plus_w = transitions.ra_look_up(t+1+ad,st_w,ra_w,d_w,par)    # wife            

                # income
                if look_up:
                    pens_h = transitions.pension_look_up_c(d_h,d_w,t+1,ad,1,st_h,st_w,ra_h,ra_w,par)    # husband
                else:
                    pass
                    # pens_h = transitions.pension(t+1,1,st_h,ra_h,a,par)             # husband                    
                
                inc_no_shock = Ra[:] + pens_h
                inc = transitions.labor_look_up_c(d_h,d_w,t+1,ad,0,st_h,st_w,ra_h,ra_w,par)             # wife
                w = par.xi_women_w            

            # c. husband working, wife retire
            elif d_h == 1 and d_w == 0:                   
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_plus_h = transitions.ra_look_up(t+1,st_h,ra_h,d_h,par)       # husband            
                ra_plus_w = transitions.ra_look_up(t+1+ad,st_w,ra_w,d_w,par)    # wife            
    
                # income
                if look_up:
                    pens_w = transitions.pension_look_up_c(d_h,d_w,t+1,ad,0,st_h,st_w,ra_h,ra_w,par)    # wife
                else:
                    pass
                    # pens_w = transitions.pension(t+1+ad,0,st_w,ra_w,a,par)          # husband

                inc_no_shock = Ra[:] + pens_w
                inc = transitions.labor_look_up_c(d_h,d_w,t+1,ad,1,st_h,st_w,ra_h,ra_w,par)             # husband  
                w = par.xi_men_w

            # d. both work
            elif d_h == 1 and d_w == 1:                   
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_plus_h = transitions.ra_look_up(t+1,st_h,ra_h,1,par)         # husband            
                ra_plus_w = transitions.ra_look_up(t+1+ad,st_w,ra_w,1,par)      # wife            

                # income 
                inc_no_shock = Ra[:]                                            # no pension
                inc = transitions.labor_look_up_c(d_h,d_w,t+1,ad,0,st_h,st_w,ra_h,ra_w,par)             # joint labor income
                w = par.w_corr             

            # e. interpolate/integrate   
            v_plus_raw_c,avg_marg_u_plus_c = shocks_GH(t,inc_no_shock,inc,w,
                                                       c[ra_plus_h,ra_plus_w],m[ra_plus_h,ra_plus_w],v[ra_plus_h,ra_plus_w],
                                                       par,d_plus)
    
            # f. indices to look up
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
    """ compute v_plus_raw and avg_marg_u_plus using GaussHermite integration if necessary
    
    Args:
        t (int): model time
        inc_no_shock (numpy.ndarray): income with no shocks
        inc (numpy.ndarray): income with shocks (the GH nodes have been multiplied to the income)
        w (numpy.ndarray): GH weights
        c (numpy.ndarray): next period consumption solution
        m (numpy.ndarray): next period wealth solution
        v (numpy.ndarray): next period value solution
        c_plus_interp (numpy.ndarray): empty container for interpolation of c_plus
        v_plus_interp (numpy.ndarray): empty container for interpolation of v_plus
        par (class): parameters
        d_plus (numpy.ndarray): choice set tomorrow

    Returns:
        v_plus_raw,avg_marg_u_plus (tuple)
    """    
    
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