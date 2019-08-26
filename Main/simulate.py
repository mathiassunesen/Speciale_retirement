import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import funs
import transitions

@njit(parallel=True)
def lifecycle(sim,sol,par,accuracy_test):
    """ simulate full life-cycle """

    # unpack (to help numba optimize)
    # solution
    c_sol = sol.c[:,:,:,:,0]
    m_sol = sol.m[:,:,:,:,0]
    v_sol = sol.v[:,:,:,:,0]        

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

    # simulation
    for t in range(par.simT):

        for i in prange(par.simN): # in parallel

            # a. check if alive
            pi = transitions.survival(t,st[i],par)            
            if  alive[t-1,i] == 0 or pi < deadP[t,i]:
                alive[t,i] = 0 # Still dead
                continue 
                          
            # b. working
            if (t < par.Tr-1 and d[t,i] == 1):
                if (t > 0):
                    m[t,i] = par.R*a[t-1,i] + transitions.income(t,st[i],par,inc_shock[t,i])

                # b.1 retirement and consumption choice
                for id in range(2):
                    c_interp[t,i,id] = linear_interp.interp_1d(m_sol[t,st[i],:,id],c_sol[t,st[i],:,id],m[t,i])
                    v_interp[t,i,id] = linear_interp.interp_1d(m_sol[t,st[i],:,id],v_sol[t,st[i],:,id],m[t,i])
                
                prob = funs.logsum_vec(v_interp[t,i,:],par)[1]
                probs[t+1,i] = prob[0,0] # save the retirement probability
            
                if (prob[0,0] > unif[t,i]): # if prob of retiring exceeds threshold
                    d[t+1,i] = 0 # retire
                    c[t,i] = c_interp[t,i,0] # optimal consumption
                    
                    #if transitions.elig(st[i]) == 0:
                    if transitions.state_translate(st[i],'elig',par) == 0:
                        ret_age[i] = 2 # no erp
                    elif transitions.age(t+1) >= 62:
                        ret_age[i] = 0
                    elif 60 <= transitions.age(t+1) <= 61:
                        ret_age[i] = 1
                    else:
                        ret_age[i] = 2

                else:
                    d[t+1,i] = 1 # work
                    c[t,i] = c_interp[t,i,1] # optimal consumption 

            # c. retired
            else:            
                m[t,i] = par.R*a[t-1,i] + transitions.pension(t,st[i],np.array([a[t-1,i]]),ret_age[i],par) # pass a as array, since otherwise pension does not work
                c[t,i] = linear_interp.interp_1d(m_sol[t,st[i],:,0],c_sol[t,st[i],:,0],m[t,i]) # optimal consumption
                if (t < par.simT-1): # if not last period
                    d[t+1,i] = 0 # still retired                
                    probs[t+1,i] = 0 # set retirement probability to 0 if already retired                        

            # d. update end of period wealth
            a[t,i] = m[t,i]-c[t,i]

            # e. accuracy
            if (accuracy_test and t < par.simT-1): # not last period
                if (1e-6 < c[t,i] < m[t,i] - 1e-6): # inner solution
                    
                    # finding rhs
                    Ra = par.R*a[t,i]
                    if d[t+1,i] == 0: # retired
                        m_plus = Ra + transitions.pension(t+1,st[i],np.array([a[t,i]]),ret_age[i],par)
                        c_plus = linear_interp.interp_1d(m_sol[t+1,st[i],:,0],c_sol[t+1,st[i],:,0],m_plus)
                        marg_u_plus = utility.marg_func(c_plus,par)
                        rhs = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 

                    else: # working   
                        if transitions.state_translate(st[i],'male',par) == 1:
                            w = par.xi_men_w
                            xi = par.xi_men
                        else:
                            w = par.xi_women_w
                            xi = par.xi_women  

                        avg_marg_u_plus = 0
                        c_plus = np.zeros((1,2))
                        v_plus = np.zeros((1,2))
                        for i in range(len(xi)):
                            m_plus = Ra + transitions.income(t+1,st[i],par,xi[i]) # m_plus is next period resources therefore income(t+1)

                            # 1. interpolate and logsum
                            for id in range(2): # in parallel
                                c_plus[:,id] = linear_interp.interp_1d(m_sol[t+1,st[i],:,id],c_sol[t+1,st[i],:,id],m_plus)
                                v_plus[:,id] = linear_interp.interp_1d(m_sol[t+1,st[i],:,id],v_sol[t+1,st[i],:,id],m_plus)

                            logsum,prob = funs.logsum_vec(v_plus,par)
                            logsum = logsum[:,0]
                            prob = prob[:,0]

                            # 2. integrate out shocks
                            marg_u_plus = prob*utility.marg_func(c_plus[:,0],par) + (1-prob)*utility.marg_func(c_plus[:,1],par)
                            avg_marg_u_plus += w[i]*marg_u_plus

                        # c. store rhs
                        rhs = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)

                    # lhs and euler
                    lhs = utility.marg_func(c[t,i],par)
                    euler[t,i] = lhs - rhs                


# @njit(parallel=True)
# def lifecycle_vec(sim,sol,par,accuracy_test):
#     """ simulate full life-cycle """

#     # unpack (to help numba optimize)
#     # solution
#     c_sol = sol.c[:,:,:,:,0]
#     m_sol = sol.m[:,:,:,:,0]
#     v_sol = sol.v[:,:,:,:,0]        

#     # simulation
#     c = sim.c
#     m = sim.m
#     a = sim.a
#     d = sim.d

#     # dummies and probabilities
#     alive = sim.alive
#     probs = sim.probs
#     ret_age = sim.ret_age

#     # interpolation
#     c_interp = sim.c_interp
#     v_interp = sim.v_interp  
#     euler = sim.euler                       

#     # random shocks
#     unif = sim.unif
#     deadP = sim.deadP
#     inc_shock = sim.inc_shock

#     # states
#     st = sim.states

#     # simulation
#     for t in range(par.simT):
#         for i in prange(par.simN): # in parallel

#             # a. check if alive
#             pi = transitions.survival(t,st[i],par)            
#             if alive[t-1,i] == 0 or pi < deadP[t,i]:
#                alive[t,i] = 0 # Still dead
#                continue 

#             # b. m for working
#             if (t < par.Tr-1 and d[t,i] == 1):
#                 if (t > 0):
#                     m[t,i] = par.R*a[t-1,i] + transitions.income(t,st[i],par,inc_shock[t,i])

#             # c. m for retired
#             else:            
#                 m[t,i] = par.R*a[t-1,i] + transitions.pension(t,st[i],np.array([a[t-1,i]]),ret_age[i],par) # pass a as array, since otherwise pension does not work                    

#         # interpolate
#         prep = linear_interp.interp_1d_prep(par.simN)
#         for ist in range(len(np.unique(st))):
#             for id in range(2):
#                 linear_interp.interp_1d_vec_mon(prep,m_sol[t,ist,:,id],c_sol[t,ist,:,id],m[t,st==ist],c_interp[t,st==ist,id])
#                 linear_interp.interp_1d_vec_mon_rep(prep,m_sol[t,ist,:,id],v_sol[t,ist,:,id],m[t,st==ist],v_interp[t,st==ist,id])                
                
#                 prob = funs.logsum_vec(v_interp[t,i,:],par)[1]
#                 probs[t+1,i] = prob[0,0] # save the retirement probability
            
#                 if (prob[0,0] > unif[t,i]): # if prob of retiring exceeds threshold
#                     d[t+1,i] = 0 # retire
#                     c[t,i] = c_interp[t,i,0] # optimal consumption
                    
#                     #if transitions.elig(st[i]) == 0:
#                     if transitions.state_translate(st[i],'elig',par) == 0:
#                         ret_age[i] = 2 # no erp
#                     elif transitions.age(t+1) >= 62:
#                         ret_age[i] = 0
#                     elif 60 <= transitions.age(t+1) <= 61:
#                         ret_age[i] = 1
#                     else:
#                         ret_age[i] = 2

#                 else:
#                     d[t+1,i] = 1 # work
#                     c[t,i] = c_interp[t,i,1] # optimal consumption 

#             # c. retired
#             else:            
#                 m[t,i] = par.R*a[t-1,i] + transitions.pension(t,st[i],np.array([a[t-1,i]]),ret_age[i],par) # pass a as array, since otherwise pension does not work
#                 c[t,i] = linear_interp.interp_1d(m_sol[t,st[i],:,0],c_sol[t,st[i],:,0],m[t,i]) # optimal consumption
#                 if (t < par.simT-1): # if not last period
#                     d[t+1,i] = 0 # still retired                
#                     probs[t+1,i] = 0 # set retirement probability to 0 if already retired                        

#             # d. update end of period wealth
#             a[t,i] = m[t,i]-c[t,i]

#             # e. accuracy
#             if (accuracy_test and t < par.simT-1): # not last period
#                 if (1e-6 < c[t,i] < m[t,i] - 1e-6): # inner solution
                    
#                     # finding rhs
#                     Ra = par.R*a[t,i]
#                     if d[t+1,i] == 0: # retired
#                         m_plus = Ra + transitions.pension(t+1,st[i],np.array([a[t,i]]),ret_age[i],par)
#                         c_plus = linear_interp.interp_1d(m_sol[t+1,st[i],:,0],c_sol[t+1,st[i],:,0],m_plus)
#                         marg_u_plus = utility.marg_func(c_plus,par)
#                         rhs = par.beta*(par.R*pi*marg_u_plus + (1-pi)*par.gamma) 

#                     else: # working   
#                         if transitions.state_translate(st[i],'male',par) == 1:
#                             w = par.xi_men_w
#                             xi = par.xi_men
#                         else:
#                             w = par.xi_women_w
#                             xi = par.xi_women  

#                         avg_marg_u_plus = 0
#                         c_plus = np.zeros((1,2))
#                         v_plus = np.zeros((1,2))
#                         for i in range(len(xi)):
#                             m_plus = Ra + transitions.income(t+1,st[i],par,xi[i]) # m_plus is next period resources therefore income(t+1)

#                             # 1. interpolate and logsum
#                             for id in range(2): # in parallel
#                                 c_plus[:,id] = linear_interp.interp_1d(m_sol[t+1,st[i],:,id],c_sol[t+1,st[i],:,id],m_plus)
#                                 v_plus[:,id] = linear_interp.interp_1d(m_sol[t+1,st[i],:,id],v_sol[t+1,st[i],:,id],m_plus)

#                             logsum,prob = funs.logsum_vec(v_plus,par)
#                             logsum = logsum[:,0]
#                             prob = prob[:,0]

#                             # 2. integrate out shocks
#                             marg_u_plus = prob*utility.marg_func(c_plus[:,0],par) + (1-prob)*utility.marg_func(c_plus[:,1],par)
#                             avg_marg_u_plus += w[i]*marg_u_plus

#                         # c. store rhs
#                         rhs = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)

#                     # lhs and euler
#                     lhs = utility.marg_func(c[t,i],par)
#                     euler[t,i] = lhs - rhs  

