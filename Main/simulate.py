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
    suvP = sim.suvP
 

    for t in prange(par.simT):
        for i in range(par.simN): # in parallel

            if  alive[t-1,i] == 0 or par.survival_probs[t] < suvP[t,i]:
                alive[t,i] = 0 # Still dead
                continue 
                          
            
            # a. states
            if t == 0: # initialize
                m[t,i] = 10
            elif t < par.Tr-1: # working
                Y = transitions.income(t,par)
                m[t,i] = par.R*a[t-1,i] + d[t-1,i]*Y
            else: # forced to retire
                m[t,i] = par.R*a[t-1,i]

            # b. retirement and consumption choice
            for id in range(2):
                c_interp[t,i,id] = linear_interp.interp_1d(sol.m[t,:,id],sol.c[t,:,id],m[t,i])
                v_interp[t,i,id] = linear_interp.interp_1d(sol.m[t,:,id],sol.v[t,:,id],m[t,i])

            logsum,prob = funs.logsum_vec(v_interp[t,i,:].reshape(1,2),par.sigma_eta)
            prob = prob[0] # unpack it. this is strange!!!
            if (t >= par.Tr-1 or d[t-1,i] == 0 or prob[0] > unif[t,i]): # if forced to retire, is retired, prob of retiring exceeds threshold
                d[t,i] = 0 # retire
                c[t,i] = c_interp[t,i,0]
            else:
                d[t,i] = 1 # work
                c[t,i] = c_interp[t,i,1]

            # c. update post decision
            a[t,i] = m[t,i]-c[t,i]