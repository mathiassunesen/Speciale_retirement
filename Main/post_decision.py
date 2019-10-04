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
def compute(t,ad,ma,st,ra,d_lst,sol,par):
    """ compute post decision for singles""" 

    # unpack solution
    c = sol.c[t+1,ad,ma,st]
    m = sol.m[t+1,ad,ma,st]
    v = sol.v[t+1,ad,ma,st]

    # unpack rest
    c_plus_interp = sol.c_plus_interp[:,:]
    v_plus_interp = sol.v_plus_interp[:,:]  
    q = sol.q[:,:]
    v_plus_raw = sol.v_plus_raw[:,:]
    pi = transitions.survival_look_up(t,ma,par)
       
    # loop over the choices
    for d in d_lst:

        # prep
        a = par.grid_a
        Ra = par.R*a      

        # a. choose to retire
        if d == 0:
            # next period income
            inc_no_shock = Ra + transitions.pension_look_up(t+1,ma,st,ra,par)
            w = np.array([0.0]) # no integration
            inc = w[:]               

            # next period choice set and retirement age 
            d_plus = transitions.d_plus(t,d,par)
            ra_plus = transitions.ra_look_up(t+1,st,ra,d,par) # if choosing to retire at t means retirement at t+1           
        
        # b. choose to work
        elif d == 1:
            # next period choice set and retirement age 
            d_plus = transitions.d_plus(t,d,par)
            ra_plus = transitions.ra_look_up(t+1,st,ra,d,par) # if choosing to work at t means soonest retirement at t+2

            # next period income
            inc_no_shock = Ra[:]
            inc = transitions.labor_look_up(t+1,ma,st,par)  

            # weights
            if ma == 1:
                w = par.xi_men_w
            elif ma == 0:
                w = par.xi_women_w                                             

        # c. integration            
        vp_raw,avg_marg_u_plus = shocks_GH(t,inc_no_shock,inc,w,c[ra_plus],m[ra_plus],v[ra_plus],
                                           c_plus_interp,v_plus_interp,par,d_plus)

        # d. store results
        v_plus_raw[d,:] = vp_raw[:]
        q[d,:] = par.beta*(par.R*pi*avg_marg_u_plus[:] + (1-pi)*par.gamma)   


###############################
### Functions for couples #####
###############################
@njit(parallel=True)
def compute_c(t,ad,st_h,st_w,d,sol,par,retirement):
    """ compute post decision for couples""" 

    # unpack solution
    ad_idx = ad+par.ad_min
    c = sol.c[t+1,ad_idx,st_h,st_w]
    m = sol.m[t+1,ad_idx,st_h,st_w]
    v = sol.v[t+1,ad_idx,st_h,st_w]  

    # unpack rest
    c_plus_interp = sol.c_plus_interp[:,:]
    v_plus_interp = sol.v_plus_interp[:,:]  
    q = sol.q[:,:]
    v_plus_raw = sol.v_plus_raw[:,:]
    pi_w,pi_h = transitions.survival_look_up_c(t,ad,par)         
       
    # loop over the choices
    for idx in d:

        # prep
        a = par.grid_a
        Ra = par.R*a

        # a. both retire
        if idx == 0:
            # choice set tomorrow and retirement age
            d_plus = transitions.d_plus_c(t,ad,idx,par)
            erp = 0            
            # d_plus = np.array([0])  # retirement is absorbing for both
            # erp = 0

            # income
            pens_h = transitions.pension_look_up_c(t+1,1,ad,st_h,retirement[1],par) # husband
            pens_w = transitions.pension_look_up_c(t+1,0,ad,st_w,retirement[1],par) # wife
            inc_no_shock = Ra[:] + pens_h[:] + pens_w[:]
            w = np.array([0.0]) # no integration
            inc = w[:]

        # b. husband retire, wife working
        elif idx == 1:                  
            # choice set tomorrow and retirement age
            d_plus = transitions.d_plus_c(t,ad,idx,par)
            erp = 0            
            # if t+1+ad == par.Tr-1:  # wife forced to retire next period
            #     d_plus = np.array([0,0])
            # else:
            #     d_plus = np.array([0,1]) 
            # erp = 0

            # income
            pens_h = transitions.pension_look_up_c(t+1,1,ad,st_h,retirement[1],par) # husband
            inc_no_shock = Ra[:] + pens_h[:]
            inc = transitions.labor_look_up_c(idx,t+1,ad,st_h,st_w,par)             # wife
            w = par.xi_women_w            

        # c. husband working, wife retire
        elif idx == 2:                   
            # choice set tomorrow and retirement age
            d_plus = transitions.d_plus_c(t,ad,idx,par)
            erp = 0            
            # if t+1 == par.Tr-1:     # husband forced retire next period
            #     d_plus = np.array([0,0])
            # else:
            #     d_plus = np.array([0,2])
            # erp = 0

            # income
            pens_w = transitions.pension_look_up_c(t+1,0,ad,st_w,retirement[1],par) # wife
            inc_no_shock = Ra[:] + pens_w[:]
            inc = transitions.labor_look_up_c(idx,t+1,ad,st_h,st_w,par)             # husband  
            w = par.xi_men_w

        # d. both work
        elif idx == 3:                   
            # choice set tomorrow and retirement age
            d_plus = transitions.d_plus_c(t,ad,idx,par)
            erp = 0
            # if ad == 0 and t+1 == par.Tr-1:
            #     d_plus = np.array([0])
            # elif ad < 0 and t+1 == par.Tr-1:
            #     d_plus = np.array([0,1])
            # elif ad > 0 and t+1+ad == par.Tr-1:
            #     d_plus = np.array([0,2])
            # else:
            #     d_plus = np.array([0,1,2,3]) 
            # erp = 0

            # income 
            inc_no_shock = Ra[:]                                        # no pension
            inc = transitions.labor_look_up_c(idx,t+1,ad,st_h,st_w,par)  # joint labor income
            w = par.w_corr             

        # e. interpolate/integrate   
        vp_raw,avg_marg_u_plus = shocks_GH(t,inc_no_shock,inc,w,c[erp],m[erp],v[erp],c_plus_interp,v_plus_interp,par,d_plus)
 
        # f. store results      
        v_plus_raw[idx,:] = vp_raw[:]
        q[idx,:] = par.beta*(par.R*(pi_h*pi_w*avg_marg_u_plus[:]) + (1-pi_h)*(1-pi_w)*par.gamma)


###############################
###       Integration     #####
###############################
@njit(parallel=True)
def shocks_GH(t,inc_no_shock,inc,w,c,m,v,c_plus_interp,v_plus_interp,par,d_plus):                          
    """ GH-integration to calculate "q" and "v_plus_raw for singles".
        
    Args:

        t=time,
        Ra=R*grid_a, 
        inc=income multiplied with GH nodes, 
        w=weights,
        c,m,v=solution arrays,
        c_plus_interp,v_plus_interp=interpolation arrays,
        par=parameters,
        d=retirement choices,
        only_v=if True only compute vp_raw"""   
    
    # a. initialize
    vp_raw = np.zeros(inc_no_shock.size)
    avg_marg_u_plus = np.zeros(inc_no_shock.size)
    prep = linear_interp.interp_1d_prep(len(inc_no_shock))    # save the position of numbers to speed up interpolation

    # b. loop over GH-nodes
    for i in range(len(w)):
        m_plus = inc_no_shock + inc[i]

        # 1. interpolate
        for d in d_plus:
            linear_interp.interp_1d_vec_mon(prep,m[d,:],c[d,:],m_plus,c_plus_interp[d,:])
            linear_interp.interp_1d_vec_mon_rep(prep,m[d,:],v[d,:],m_plus,v_plus_interp[d,:])         

        # 2. logsum and v_plus_raw
        if len(d_plus) == 1:     # no taste shocks
            vp_raw = v_plus_interp[d_plus[0],:]
            avg_marg_u_plus = utility.marg_func(c_plus_interp[d_plus[0],:],par)

        elif len(d_plus) == 2:   # taste shocks
            logsum,prob = funs.logsum2(v_plus_interp[d_plus,:],par)
            vp_raw += w[i]*logsum[0,:]
            marg_u_plus = prob[d_plus[0],:]*utility.marg_func(c_plus_interp[d_plus[0],:],par) + (1-prob[d_plus[0],:])*utility.marg_func(c_plus_interp[d_plus[1],:],par)
            avg_marg_u_plus += w[i]*marg_u_plus    

        elif len(d_plus) == 4:   # both are working
            logsum,prob = funs.logsum4(v_plus_interp[d_plus,:],par)
            vp_raw += w[i]*logsum[0,:]
            marg_u_plus = (prob[0,:]*utility.marg_func(c_plus_interp[0,:],par) + 
                           prob[1,:]*utility.marg_func(c_plus_interp[1,:],par) + 
                           prob[2,:]*utility.marg_func(c_plus_interp[2,:],par) +
                           prob[3,:]*utility.marg_func(c_plus_interp[3,:],par))  
            avg_marg_u_plus += w[i]*marg_u_plus        

    # c. return results
    return vp_raw,avg_marg_u_plus          