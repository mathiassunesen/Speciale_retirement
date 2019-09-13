# global modules
import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp

# local modules
import utility
import funs
import transitions


@njit(parallel=True)
def compute_singles(t,st,ad,d,sol,par,retirement):
    """ compute post decision for singles
        
        Args:

            t=time, st=state, ad=age difference
            d=retirement choice
            sol=solution, par=parameters
            retirement=list with info about erp status""" 

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
        v_plus_raw[:,id] = vp_raw[:]
        q[:,id] = par.beta*(par.R*pi*avg_marg_u_plus[:] + (1-pi)*par.gamma)                  


@njit(parallel=True)
def shocks_GH(t,st,Ra,w,xi,c,m,v,c_plus_interp,v_plus_interp,par):                          
    """ GH-integration to calculate "q" and "v_plus_raw".
        
    Args:

        t=time, st=state,
        Ra=R*grid_a, w=weights, xi=nodes,
        c,m,v=solution arrays,
        c_plus_interp,v_plus_interp=interpolation arrays,
        par=parameters"""   
    
    # a. initialize
    vp_raw = np.zeros(Ra.size)
    avg_marg_u_plus = np.zeros(Ra.size)

    # b. loop over GH-nodes
    for i in range(len(xi)):
        m_plus = Ra + transitions.income(t+1,st,par,xi[i])  # m_plus is next period resources therefore income(t+1)

        # 1. interpolate and logsum
        prep = linear_interp.interp_1d_prep(len(m_plus))    # save the position of numbers to speed up interpolation
        for id in range(2):
            linear_interp.interp_1d_vec_mon(prep,m[:,id],c[:,id],m_plus,c_plus_interp[:,id])
            linear_interp.interp_1d_vec_mon_rep(prep,m[:,id],v[:,id],m_plus,v_plus_interp[:,id])         

        logsum,prob = funs.logsum_vec(v_plus_interp,par)
        prob0 = prob[:,0]

        # 2. integrate out shocks
        vp_raw += w[i]*logsum[:,0] 
        marg_u_plus = prob0*utility.marg_func(c_plus_interp[:,0],par) + (1-prob0)*utility.marg_func(c_plus_interp[:,1],par)
        avg_marg_u_plus += w[i]*marg_u_plus

    # c. return results
    return vp_raw,avg_marg_u_plus  


@njit(parallel=True)
def value_of_choice(t,st,ad,d,m,c,v,v_plus_interp,sol,par,retirement):
    """ compute value-of-choice, v".
        
    Args:

        t=time, st=state, ad=age difference
        d=retirement choice
        m,c=solution on constraint (today)
        v=value-of-choice on constraint (computed here) 
        v_plus_interp=interpolation array
        sol=solution, par=parameters
        retirement=list with info about erp status"""     

    # unpack
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

            # integrate out shocks
            v_plus_raw += w[i]*logsum[:,0]
        
        # 3. value-of-choice
        v[:] = utility.func(c,d,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*a)                          