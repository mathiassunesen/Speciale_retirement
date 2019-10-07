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
def compute(t,ad,ma,st,ra,D,sol,par):
    """ compute post decision states v_plus_raw and q, which is used to solve the bellman equation
    
    Args:
        t (int): model time
        ad (int): age difference, just zero here
        ma (int): 0 if female and 1 if male
        st (int): states
        ra (int): retirement status
        D (numpy.ndarray): choice set
        sol (class): solution
        par (class): parameters

    Returns:
        stores v_plus_raw and q in sol
    """        

    # unpack solution
    c_plus = sol.c[t+1,ad,ma,st]
    m_plus = sol.m[t+1,ad,ma,st]
    v_plus = sol.v[t+1,ad,ma,st]

    # unpack rest
    c_plus_interp = sol.c_plus_interp[:,:]
    v_plus_interp = sol.v_plus_interp[:,:]  
    avg_marg_u_plus = sol.avg_marg_u_plus[t+1,ad,ma,st,ra]
    v_plus_raw = sol.v_plus_raw[t+1,ad,ma,st,ra]
       
    # loop over the choices
    for d in D:

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
            ra_plus = transitions.ra_look_up(t+1,st,ra,d,par)           
        
        # b. choose to work
        elif d == 1:
            # next period choice set and retirement age 
            d_plus = transitions.d_plus(t,d,par)
            ra_plus = transitions.ra_look_up(t+1,st,ra,d,par)

            # next period income
            inc_no_shock = Ra[:]
            inc = transitions.labor_look_up(t+1,ma,st,par)  

            # weights
            if ma == 1:
                w = par.xi_men_w
            elif ma == 0:
                w = par.xi_women_w                                             

        # c. integration and store results         
        v_plus_raw[d],avg_marg_u_plus[d] = shocks_GH(t,inc_no_shock,inc,w,c_plus[ra_plus],m_plus[ra_plus],v_plus[ra_plus],
                                                     c_plus_interp,v_plus_interp,par,d_plus)

        # # d. store results
        # v_plus_raw[d,:] = vp_raw[:]
        # q[d,:] = par.beta*(par.R*pi*avg_marg_u_plus[:] + (1-pi)*par.gamma)   


###############################
### Functions for couples #####
###############################
@njit(parallel=True)
def compute_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,sol,par,single_sol):
    """ compute post decision for couples""" 

    # unpack solution
    ad_min = par.ad_min
    ad_idx = ad+ad_min
    c_plus = sol.c[t+1,ad_idx,st_h,st_w]
    m_plus = sol.m[t+1,ad_idx,st_h,st_w]
    v_plus = sol.v[t+1,ad_idx,st_h,st_w]  

    # unpack rest
    c_plus_interp = sol.c_plus_interp[:,:]
    v_plus_interp = sol.v_plus_interp[:,:]  
    avg_marg_u_plus = sol.avg_marg_u_plus[t+1,ad_idx,st_h,st_w,ra_h,ra_w]
    v_plus_raw = sol.v_plus_raw[t+1,ad_idx,st_h,st_w,ra_h,ra_w]    
    pi_h,pi_w = transitions.survival_look_up_c(t+1,ad,par)  # it is pi_plus, but we save the plus
       
    # prep
    a = par.grid_a
    Ra = par.R*a

    # loop over the choices
    for d_h in D_h:
        for d_w in D_w:

            # a. both retire
            if d_h == 0 and d_w == 0:
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_pH = transitions.ra_look_up(t+1,st_h,ra_h,d_h,par)           # husband            
                ra_pW = transitions.ra_look_up(t+1+ad,st_w,ra_w,d_w,par)        # wife

                # income
                pens_h = transitions.pension_look_up_c(t+1,1,ad,st_h,ra_h,par)  # husband
                pens_w = transitions.pension_look_up_c(t+1,0,ad,st_w,ra_w,par)  # wife
                inc_no_shock = Ra[:] + pens_h[:] + pens_w[:]
                w = np.array([0.0]) # no integration
                inc = w[:]

            # b. husband retire, wife working
            elif d_h == 0 and d_w == 1:                  
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_pH = transitions.ra_look_up(t+1,st_h,ra_h,d_h,par)           # husband            
                ra_pW = transitions.ra_look_up(t+1+ad,st_w,ra_w,d_w,par)        # wife            

                # income
                pens_h = transitions.pension_look_up_c(t+1,1,ad,st_h,ra_h,par)  # husband
                inc_no_shock = Ra[:] + pens_h[:]
                inc = transitions.labor_look_up_c(d_h,d_w,t+1,ad,st_h,st_w,par) # wife
                w = par.xi_women_w            

            # c. husband working, wife retire
            elif d_h == 1 and d_w == 0:                   
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_pH = transitions.ra_look_up(t+1,st_h,ra_h,d_h,par)           # husband            
                ra_pW = transitions.ra_look_up(t+1+ad,st_w,ra_w,d_w,par)        # wife            
    
                # income
                pens_w = transitions.pension_look_up_c(t+1,0,ad,st_w,ra_w,par)  # wife
                inc_no_shock = Ra[:] + pens_w[:]
                inc = transitions.labor_look_up_c(d_h,d_w,t+1,ad,st_h,st_w,par) # husband  
                w = par.xi_men_w

            # d. both work
            elif d_h == 1 and d_w == 1:                   
                # choice set tomorrow and retirement age
                d_plus = transitions.d_plus_c(t,ad,d_h,d_w,par)                 # joint index
                ra_pH = transitions.ra_look_up(t+1,st_h,ra_h,1,par)             # husband            
                ra_pW = transitions.ra_look_up(t+1+ad,st_w,ra_w,1,par)          # wife            

                # income 
                inc_no_shock = Ra[:]                                            # no pension
                inc = transitions.labor_look_up_c(d_h,d_w,t+1,ad,st_h,st_w,par) # joint labor income
                w = par.w_corr             

            # e. integration  
            v_plus_raw_joint,avg_marg_u_plus_joint = shocks_GH(t,inc_no_shock,inc,w,c_plus[ra_pH,ra_pW],m_plus[ra_pH,ra_pW],v_plus[ra_pH,ra_pW],
                                                               c_plus_interp,v_plus_interp,par,d_plus)

            # f. look up in single solution

            # 1. indices
            d = transitions.d_c(d_h,d_w)                    # joint index
            d_plus_h = transitions.d_plus_int(t,d_h,par)    # choice tomorrow
            d_plus_w = transitions.d_plus_int(t,d_w,par)    # choice tomorrow

            # 2. look up, husband
            m_raw_h = single_sol.m_raw[t+ad_min,0,1,st_h,ra_h,d_plus_h]                     # ad=0 and male=1
            v_plus_raw_h = single_sol.v_plus_raw[t+ad_min,0,1,st_h,ra_h,d_plus_h]           # ad=0 and male=1
            avg_marg_u_plus_h = single_sol.avg_marg_u_plus[t+ad_min,0,1,st_h,ra_h,d_plus_h] # ad=0 and male=1

            # 3. look up, wife
            m_raw_w = np.zeros(m_raw_h.shape)                       # initalize          
            v_plus_raw_w = np.zeros(v_plus_raw_h.shape)             # initialize            
            avg_marg_u_plus_w = np.zeros(avg_marg_u_plus_h.shape)   # initialize            
            if t+ad < par.T:    # wife alive
                m_raw_w[:] = single_sol.m_raw[t+ad_idx,0,0,st_w,ra_w,d_plus_w]                      # ad=0 and male=0
                v_plus_raw_w[:] = single_sol.v[t+ad_idx,0,0,st_w,ra_w,d_plus_w]                     # ad=0 and male=0             
                avg_marg_u_plus_w[:] = single_sol.avg_marg_u_plus[t+ad_idx,0,0,st_w,ra_w,d_plus_w]  # ad=0 and male=0

            # 4. write to same grid
            linear_interp.interp_1d_vec(m_raw_h,v_plus_raw_h,m_plus,c_plus_interp[d,:])
            linear_interp.interp_1d_vec_mon_rep(prep,m[d,:],v[d,:],m_plus,v_plus_interp[d,:])              


            # g. store results           
            v_plus_raw[d] = par.beta*(pi_h*pi_w*v_plus_raw_joint +
                                     (1-pi_w)*pi_h*v_plus_raw_h + 
                                     (1-pi_h)*pi_w*v_plus_raw_w + 
                                     (1-pi_h)*(1-pi_w)*par.gamma*a)
            
            avg_marg_u_plus[d] = par.beta*(par.R*(pi_h*pi_w*avg_marg_u_plus_joint +
                                                 (1-pi_w)*pi_h*avg_marg_u_plus_h + 
                                                 (1-pi_h)*pi_w*avg_marg_u_plus_h) + 
                                                 (1-pi_w)*(1-pi_h)*par.gamma*a) 


###############################
###       Integration     #####
###############################
@njit(parallel=True)
def shocks_GH(t,inc_no_shock,inc,w,c,m,v,c_plus_interp,v_plus_interp,par,d_plus):     
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
    v_plus_raw = np.zeros(inc_no_shock.size)
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
            v_plus_raw = v_plus_interp[d_plus[0],:]
            avg_marg_u_plus = utility.marg_func(c_plus_interp[d_plus[0],:],par)

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