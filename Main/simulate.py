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
    m = sim.m
    c = sim.c
    a = sim.a
    d = sim.d
    alive = sim.alive
    c_interp = sim.c_interp
    v_interp = sim.v_interp                         
    unif = sim.unif
    deadP = sim.deadP

    for t in range(par.simT):
        for i in prange(par.simN): # in parallel

            # a. check if alive
            if  alive[t-1,i] == 0 or par.survival_probs[t] < deadP[t,i]:
                alive[t,i] = 0 # Still dead
                continue 
                          
            # working
            if (t < par.Tr-1 and d[t,i] == 1):
                if (t > 0):
                    m[t,i] = par.R*a[t-1,i] + transitions.income(t,par)

                # retirement and consumption choice
                for id in range(2):
                    c_interp[t,i,id] = linear_interp.interp_1d(sol.m[t,:,id],sol.c[t,:,id],m[t,i])
                    v_interp[t,i,id] = linear_interp.interp_1d(sol.m[t,:,id],sol.v[t,:,id],m[t,i])
                
                logsum,prob = funs.logsum_vec(v_interp[t,i,:].reshape(1,2),par)
                prob = prob[0] # unpack it. this is strange!!!
            
                if (prob[0] > unif[t,i]): # if prob of retiring exceeds threshold
                    d[t+1,i] = 0 # retire
                    c[t,i] = c_interp[t,i,0]
                else:
                    d[t+1,i] = 1 # work
                    c[t,i] = c_interp[t,i,1]                                                        
            
            # retired
            else:            
                m[t,i] = par.R*a[t-1,i] + transitions.pension(t,np.array([a[t-1,i]]))
                c[t,i] = linear_interp.interp_1d(sol.m[t,:,0],sol.c[t,:,0],m[t,i])
                if (t < par.simT-1): # if not last period
                    d[t+1,i] = 0 # still retired                

            # b. update end of period wealth
            a[t,i] = m[t,i]-c[t,i]


