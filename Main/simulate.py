import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import funs
import transitions
import post_decision

@njit(parallel=True)
def lifecycle(sim,sol,par,accuracy_test):
    """ Simulate full life-cycle
        
        Args:

            sim,sol,par=lists
            accuracy_test=if True; calculate euler errors"""         

    # unpack (to help numba optimize)
    # solution
    c_sol_full = sol.c[:,:,par.age_dif[0],:,:]
    m_sol_full = sol.m[:,:,par.age_dif[0],:,:]
    v_sol_full = sol.v[:,:,par.age_dif[0],:,:]        

    # simulation
    c = sim.c
    m = sim.m
    a = sim.a
    d = sim.d

    # dummies and probabilities
    alive = sim.alive
    probs = sim.probs
    ret_age = sim.ret_age

    # interpolation
    c_interp = sim.c_interp
    v_interp = sim.v_interp  
    euler = sim.euler                       

    # random shocks
    unif = sim.unif
    deadP = sim.deadP
    inc_shock = sim.inc_shock

    # states
    st = sim.states

    # misc
    tol = par.tol

    # simulation
    for t in range(par.simT):       # loop over time
        for i in prange(par.simN):  # loop over individuals

            # a. check if alive
            pi = transitions.survival_look_up(t,st[i],par)            
            if  alive[t-1,i] == 0 or pi < deadP[t,i]:
                alive[t,i] = 0 # Still dead
                continue 

            # b unpack time and state
            m_sol = m_sol_full[t,st[i]]
            c_sol = c_sol_full[t,st[i]]
            v_sol = v_sol_full[t,st[i]]
            st_i = st[i] # specific state                
                          
            # c. working
            if (t < par.Tr-1 and d[t,i] == 1):  # if not retired
                if (t > 0): # if not 1. period (where m is initialized)

                    # update m
                    m[t,i] = par.R*a[t-1,i] + transitions.income(t,st_i,par,inc_shock[t,i])

                # retirement and consumption choice
                for id in range(2):
                    c_interp[t,i,id] = linear_interp.interp_1d(m_sol[:,id],c_sol[:,id],m[t,i])
                    v_interp[t,i,id] = linear_interp.interp_1d(m_sol[:,id],v_sol[:,id],m[t,i])
                
                prob = funs.logsum_vec(v_interp[t,i,:],par)[1]
                probs[t+1,i] = prob[0,0] # save the retirement probability
                            
                if (prob[0,0] > unif[t,i]): # if prob of retiring exceeds threshold
                    d[t+1,i] = 0 # retire
                    c[t,i] = c_interp[t,i,0] # optimal consumption
                    
                    # update retirment age
                    if transitions.state_translate(st_i,'elig',par) == 0:
                        ret_age[i] = 2 # no erp
                    elif transitions.age(t+1,par) >= 62:
                        ret_age[i] = 0
                    elif 60 <= transitions.age(t+1,par) <= 61:
                        ret_age[i] = 1
                    else:
                        ret_age[i] = 2

                else: # still working
                    d[t+1,i] = 1 # work
                    c[t,i] = c_interp[t,i,1] # optimal consumption 

            # c. retired
            else:            

                # update m and c
                m[t,i] = par.R*a[t-1,i] + transitions.pension(t,st_i,a[t-1,i],ret_age[i],par) # pass a as array, since otherwise pension does not work
                c[t,i] = linear_interp.interp_1d(m_sol[:,0],c_sol[:,0],m[t,i])
                
                # update retirement choice and probability
                if (t < par.simT-1): # if not last period
                    d[t+1,i] = 0 # still retired                
                    probs[t+1,i] = 0 # set retirement probability to 0 if already retired                        

            # d. update end of period wealth
            a[t,i] = m[t,i]-c[t,i]

            # e. accuracy
            if (accuracy_test and t < par.simT-1): # not last period
                if (tol < c[t,i] < m[t,i] - tol): # inner solution
                    euler_error(t,i,st_i,pi,ret_age,c,a,d,c_sol,m_sol,v_sol,par,euler) # calculate euler errors


@njit(parallel=True)
def euler_error(t,i,st_i,pi,ret_age,c,a,d,c_sol,m_sol,v_sol,par,euler):
    """ Calculate euler errors 
    
        Args:
            
            t=time, i=individual, st_i=specific state,
            pi=survival probs, ret_age=retirement age
            c=consumption, a=assets, d=retirement choice,
            c_sol,m_sol,v_sol=solution,
            par=parameters,
            euler=to store euler errors"""

    # a. finding rhs
    Ra = par.R*a[t,i]
    if d[t+1,i] == 0: # retired
        
        m_plus = Ra + transitions.pension(t+1,st_i,a[t,i],ret_age[i],par)
        c_plus = linear_interp.interp_1d(m_sol[:,0],c_sol[:,0],m_plus)
        marg_u_plus = utility.marg_func(c_plus,par)
        rhs = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 

    else: # working 

        # 1. prepare for GH-integration  
        if transitions.state_translate(st_i,'male',par) == 1:
            w = par.xi_men_w
            xi = par.xi_men
        else:
            w = par.xi_women_w
            xi = par.xi_women  

        # 2. GH integration
        c_plus = np.zeros((1,2))
        v_plus = np.zeros((1,2))                        
        avg_marg_u_plus = post_decision.shocks_GH(t,st_i,Ra,w,xi,c_sol,m_sol,v_sol,c_plus,v_plus,par)[1]
                      
        # 3. store rhs
        rhs = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)

    # b. lhs and euler
    lhs = utility.marg_func(c[t,i],par)
    euler[t,i] = lhs - rhs                