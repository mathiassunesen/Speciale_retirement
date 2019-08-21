import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import funs
import transitions

@njit(parallel=True)
def lifecycle(sim,sol,par):
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

    # random shocks
    unif = sim.unif
    deadP = sim.deadP

    # states
    st = sim.states

    # simulation
    for t in range(par.simT):

        for i in prange(par.simN): # in parallel

            # a. check if alive
            survival_probs = transitions.survival(t,st[i],par)            
            if  alive[t-1,i] == 0 or survival_probs < deadP[t,i]:
                alive[t,i] = 0 # Still dead
                continue 
                          
            # b. working
            if (t < par.Tr-1 and d[t,i] == 1):
                if (t > 0):
                    m[t,i] = par.R*a[t-1,i] + transitions.income(t,st[i],par)

                # b.1 retirement and consumption choice
                for id in range(2):
                    c_interp[t,i,id] = linear_interp.interp_1d(m_sol[t,st[i],:,id],c_sol[t,st[i],:,id],m[t,i])
                    v_interp[t,i,id] = linear_interp.interp_1d(m_sol[t,st[i],:,id],v_sol[t,st[i],:,id],m[t,i])
                
                prob = funs.logsum_vec(v_interp[t,i,:],par)[1]
                probs[t+1,i] = prob[0,0] # save the retirement probability
            
                if (prob[0,0] > unif[t,i]): # if prob of retiring exceeds threshold
                    d[t+1,i] = 0 # retire
                    c[t,i] = c_interp[t,i,0]
                    
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
                    c[t,i] = c_interp[t,i,1]                                                        
            
            # c. retired
            else:            
                m[t,i] = par.R*a[t-1,i] + transitions.pension(t,st[i],np.array([a[t-1,i]]),ret_age[i],par)
                c[t,i] = linear_interp.interp_1d(m_sol[t,st[i],:,0],c_sol[t,st[i],:,0],m[t,i])
                if (t < par.simT-1): # if not last period
                    d[t+1,i] = 0 # still retired                
                    probs[t+1,i] = 0 # set retirement probability to 0 if already retired

            # d. update end of period wealth
            a[t,i] = m[t,i]-c[t,i]


