import numpy as np
from numba import njit, prange
import time
from scipy.interpolate import RegularGridInterpolator

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import funs
import transitions


@njit(parallel=True)
def compute_singles(t,st,ad,d,sol,par,retirement=[0,0,0]):

    # unpack solution
    if retirement[0] == 0:
        c = sol.c[t+1,st,ad]
        m = sol.m[t+1,st,ad]
        v = sol.v[t+1,st,ad]

    else: # if we have to recalculate erp status
        c = sol.c_dummy[t+1-par.dummy_t,:,:,retirement[0]-1]
        m = sol.m_dummy[t+1-par.dummy_t,:,:,retirement[0]-1]
        v = sol.v_dummy[t+1-par.dummy_t,:,:,retirement[0]-1]           

    # unpack rest
    poc = par.poc # points on constraint
    c_plus_interp = sol.c_plus_interp[poc:,:]
    v_plus_interp = sol.v_plus_interp[poc:,:]  
    q = sol.q[:,:]
    v_plus_raw = sol.v_plus_raw[:,:]
    pi = transitions.survival_look_up(t,st,par)
       
    # loop over the choices
    for id in d:

        # a. prep
        a = par.grid_a
        Ra = par.R*a      

        if id == 0: # retired

            # 1. next period resources
            m_plus = Ra + transitions.pension(t+1,st,a,retirement[1],par)        

            # 2. interpolate   
            prep = linear_interp.interp_1d_prep(len(m))    
            linear_interp.interp_1d_vec_mon(prep,m[:,id],c[:,id],m_plus,c_plus_interp[:,id])
            linear_interp.interp_1d_vec_mon_rep(prep,m[:,id],v[:,id],m_plus,v_plus_interp[:,id])  
            vp_raw = v_plus_interp[:,id]   

            # 3. next period marginal utility
            avg_marg_u_plus = utility.marg_func(c_plus_interp[:,id],par)

            # 4. reset c,m,v pointers
            if retirement[0] != 0:
                c = sol.c[t+1,st,ad]
                m = sol.m[t+1,st,ad]
                v = sol.v[t+1,st,ad]                

        elif id == 1: # working

            # 1. if necesarry, initialize solution
            if t+1 == par.Tr-2: # if forced to retire next period (which means actual retirement two periods from now due to the timing of retirement decision)
                c[:,1] = c[:,0] # initializing work solution, which is equal to retirement solution
                m[:,1] = m[:,0]
                v[:,1] = v[:,0]                   

            # 2. prepare for GH integration
            if transitions.state_translate(st,'male',par) == 1:
                w = par.xi_men_w
                xi = par.xi_men
            else:
                w = par.xi_women_w
                xi = par.xi_women

            # 3. GH integration
            vp_raw,avg_marg_u_plus = shocks_GH(t,st,Ra,w,xi,c,m,v,c_plus_interp,v_plus_interp,par)

        # b. store results
        v_plus_raw[:,id] = vp_raw
        q[:,id] = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)     


@njit(parallel=True)
def shocks_GH(t,st,Ra,w,xi,c,m,v,c_plus_interp,v_plus_interp,par):                          
    """ Performs GH-integration to calculate "q" and "v_plus_raw".
        
    Args:

        t=time, st=states,
        Ra=R*grid_a, w=GH-weights, xi=GH-nodes,
        c,m,v=solution,
        c_plus_interp,v_plus_interp=interpolation,
        par=parameters"""   
    
    # a. initialize
    vp_raw = np.zeros(Ra.size)
    avg_marg_u_plus = np.zeros(Ra.size)

    # b. loop over GH-nodes
    for i in range(len(xi)):
        m_plus = Ra + transitions.income(t+1,st,par,xi[i]) # m_plus is next period resources therefore income(t+1)

        # 1. interpolate and logsum
        if len(c_plus_interp) > 1: # vectorized
            prep = linear_interp.interp_1d_prep(len(m)) # save the position of numbers to speed up interpolation
            for id in range(2):
                linear_interp.interp_1d_vec_mon(prep,m[:,id],c[:,id],m_plus,c_plus_interp[:,id])
                linear_interp.interp_1d_vec_mon_rep(prep,m[:,id],v[:,id],m_plus,v_plus_interp[:,id])
        
        else: # loop based, to be compatible with simulate
            for id in range(2):
                c_plus_interp[0,id] = linear_interp.interp_1d(m[:,id],c[:,id],m_plus)
                v_plus_interp[0,id] = linear_interp.interp_1d(m[:,id],v[:,id],m_plus)            

        logsum,prob = funs.logsum_vec(v_plus_interp,par)
        logsum = logsum[:,0]
        prob = prob[:,0]

        # 2. integrate out shocks
        vp_raw += w[i]*logsum 
        marg_u_plus = prob*utility.marg_func(c_plus_interp[:,0],par) + (1-prob)*utility.marg_func(c_plus_interp[:,1],par)
        avg_marg_u_plus += w[i]*marg_u_plus

    # c. return results
    return vp_raw,avg_marg_u_plus  


@njit(parallel=True)
def value_of_choice(t,st,ad,d,m,c,v,sol,par,retirement):
    """ Compute value-of-choice"""

    # unpack
    # misc
    poc = par.poc
    v_plus_interp = sol.v_plus_interp[:poc,:]
    pi = transitions.survival_look_up(t,st,par)

    # next period solution
    if retirement[0] == 0 or d == 1:
        m_next = sol.m[t+1,st,ad,:,:]
        v_next = sol.v[t+1,st,ad,:,:]
    else:
        m_next = sol.m_dummy[t+1-par.dummy_t,:,:,retirement[0]-1]
        v_next = sol.v_dummy[t+1-par.dummy_t,:,:,retirement[0]-1]        
            
    # a. next period ressources and value
    a = m - c
    Ra = par.R*a

    if d == 0: # retired
        m_plus = Ra + transitions.pension(t+1,st,a,retirement[1],par)        
        linear_interp.interp_1d_vec(m_next[:,d],v_next[:,d],m_plus,v_plus_interp[:,d])
        v[:] = utility.func(c,d,st,par) + par.beta*(pi*v_plus_interp[:,d] + (1-pi)*par.gamma*a)

    else: # working

        # 1. prepare for GH integration
        if transitions.state_translate(st,'male',par) == 1:
            w = par.xi_men_w
            xi = par.xi_men
        else:
            w = par.xi_women_w
            xi = par.xi_women   

        # 2. loop over GH nodes
        v_plus_raw = np.zeros(Ra.shape)
        for i in range(len(xi)):
            m_plus = Ra + transitions.income(t+1,st,par,xi[i]) # m_plus is next period resources therefore income(t+1)

            # interpolate and logsum
            for id in range(2):
                linear_interp.interp_1d_vec(m_next[:,id],v_next[:,id],m_plus,v_plus_interp[:,id])

            logsum = funs.logsum_vec(v_plus_interp,par)[0]
            logsum = logsum[:,0]

            # integrate out shocks
            v_plus_raw += w[i]*logsum   
        
        # 3. value-of-choice
        v[:] = utility.func(c,d,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*a)                          










@njit(parallel=True)
def compute_retired(t,st,ad,sol,par,retirement=[0,0,0]):
    """ Compute the post-decision functions "q" and "v_plus_raw" if retired. 

    Args:

        t=time, st=states, ad=age difference, sol=solution, par=parameters,
        retirement=list used to recalculate erp"""      

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    if retirement[0] == 0:
        c = sol.c[t+1,st,ad,:,0]
        m = sol.m[t+1,st,ad,:,0]
        v = sol.v[t+1,st,ad,:,0]
    else:
        c = sol.c_dummy[t+1-par.dummy_t,:,0,retirement[0]-1]
        m = sol.m_dummy[t+1-par.dummy_t,:,0,retirement[0]-1]
        v = sol.v_dummy[t+1-par.dummy_t,:,0,retirement[0]-1]                
    
    c_plus_interp = sol.c_plus_interp[poc:,0]
    v_plus_interp = sol.v_plus_interp[poc:,0]  
    q = sol.q[:,0]
    v_plus_raw = sol.v_plus_raw[:,0]
    
    # a. next period ressources and value
    a = par.grid_a
    m_plus = par.R*a + transitions.pension(t+1,st,a,retirement[1],par)        

    # b. interpolate   
    prep = linear_interp.interp_1d_prep(len(m))    
    linear_interp.interp_1d_vec_mon(prep,m,c,m_plus,c_plus_interp)
    linear_interp.interp_1d_vec_mon_rep(prep,m,v,m_plus,v_plus_interp)     

    # c. next period marginal utility
    marg_u_plus = utility.marg_func(c_plus_interp,par)

    # d. store results
    pi = transitions.survival_look_up(t,st,par)
    v_plus_raw[:] = v_plus_interp
    q[:] = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 


@njit(parallel=True)
def compute_work(t,st,ad,sol,par):
    """ Compute the post-decision functions "q" and "v_plus_raw" if working. 
        This is a wrapper, which calls "shocks_GH", 
        where the main part takes place

    Args:

        t=time, st=states, sol=solution, par=parameters"""            

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t+1,st,ad,:,:] 
    m = sol.m[t+1,st,ad,:,:]
    v = sol.v[t+1,st,ad,:,:]

    if t+1 == par.Tr-2: # if forced to retire next period (which means actual retirement two periods from now due to the timing of retirement decision)
        c[:,1] = c[:,0] # initializing work solution, which is equal to retirement solution
        m[:,1] = m[:,0]
        v[:,1] = v[:,0]
                        
    c_plus_interp = sol.c_plus_interp[poc:,:]
    v_plus_interp = sol.v_plus_interp[poc:,:]
    q = sol.q[:,1]
    v_plus_raw = sol.v_plus_raw[:,1]

    # a. next period ressources and value - prepare for GH integration
    Ra = par.R*par.grid_a
    if transitions.state_translate(st,'male',par) == 1:
        w = par.xi_men_w
        xi = par.xi_men
    else:
        w = par.xi_women_w
        xi = par.xi_women

    # b. GH integration
    vp_raw,avg_marg_u_plus = shocks_GH(t,st,Ra,w,xi,c,m,v,c_plus_interp,v_plus_interp,par)

    # c. store results
    pi = transitions.survival_look_up(t,st,par)  
    v_plus_raw[:] = vp_raw
    q[:] = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)    