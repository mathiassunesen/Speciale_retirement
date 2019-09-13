# global modules
import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp

# local modules
import utility
import funs
import transitions
import post_decision


def lifecycle(sim,sol,par):
    """ Simulate full life-cycle
        
        Args:

            sim=simulation, sol=solution, par=parameters"""         

    # unpack (to help numba optimize)
    # solution
    c_sol = sol.c[:,:,par.age_dif[0],:,:]
    m_sol = sol.m[:,:,par.age_dif[0],:,:]
    v_sol = sol.v[:,:,par.age_dif[0],:,:]        

    # simulation
    c = sim.c
    m = sim.m
    a = sim.a
    d = sim.d

    # dummies and probabilities
    alive = sim.alive
    probs = sim.probs
    ret_age = sim.ret_age                          

    # states
    states = sim.states
    num_st = np.unique(states)

    # random shocks
    unif = sim.unif
    deadP = sim.deadP
    inc_shock = sim.inc_shock    

    # simulation
    for t in range(par.simT):   # loop over time

        if t > 0:
            alive[t,alive[t-1] == 0] = 0    # still dead

        for st in num_st:       # loop over states
        
            # 1. update alive
            mask_st = np.nonzero(states==st)[0]             # mask of states
            pi = transitions.survival_look_up(t,st,par)
            alive[t,mask_st[pi < deadP[t,mask_st]]] = 0     # dead

            # 2. create masks
            work = mask_st[(d[t,mask_st]==1) & (alive[t,mask_st]==1)]   # mask for state, work and alive
            ret = mask_st[(d[t,mask_st]==0) & (alive[t,mask_st]==1)]    # mask for state, retired and alive            

            # 3. update m
            if t > 0:               # m is initialized in 1. period
                if t < par.Tr-1:    # if not forced retire
                    m[t,work] = par.R*a[t-1,work] + transitions.income(t,st,par,inc_shock[t,work])

                for ra in np.unique(ret_age[ret]):  # loop over retirement age (erp status)
                    mask_ra = (ret_age[ret] == ra)
                    m[t,ret[mask_ra]] = par.R*a[t-1,ret[mask_ra]] + transitions.pension(t,st,a[t-1,ret[mask_ra]],ra,par)
                    # The above line could turn out to be problematic if a[t-1,ret[mask_ra]] is not an array (only has 1 observation)

            # 4. working
            if t < par.Tr-1 and work.size > 0:  # not forced to retire and working
    
                # a. interpolation
                prep = linear_interp.interp_1d_prep(len(work))  
                c_interp = np.nan*np.zeros((len(work),2))
                v_interp = np.nan*np.zeros((len(work),2))  
                idx = np.argsort(m[t,work]) # indices to sort back later
                m_sort = np.sort(m[t,work]) # sort m so interp is faster
                for id in range(2):
                    linear_interp.interp_1d_vec_mon(prep,m_sol[t,st,:,id],c_sol[t,st,:,id],m_sort,c_interp[:,id])
                    linear_interp.interp_1d_vec_mon_rep(prep,m_sol[t,st,:,id],v_sol[t,st,:,id],m_sort,v_interp[:,id])

                # sort back
                c_interp = c_interp[np.argsort(idx),:]
                v_interp = v_interp[np.argsort(idx),:]

                # b. retirement probabilities
                prob = funs.logsum_vec(v_interp,par)[1]
                work_choice = prob[:,0] < unif[t,work]    
                ret_choice = prob[:,0] > unif[t,work]

                # c. optimal choices for today (consumption)
                c[t,work[work_choice]] = c_interp[work_choice,1]
                c[t,work[ret_choice]] = c_interp[ret_choice,0]

                # d. optimal choices for tomorrow (retirement)
                if t < par.simT-1:
                    probs[t+1,work] = prob[:,0] # save retirement probability
                    d[t+1,work[ret_choice]] = 0
                    if t+1 >= par.Tr-1: # forced to retire
                        d[t+1,work[work_choice]] = 0
                    else:
                        d[t+1,work[work_choice]] = 1

                # e. update retirement age (erp status)
                if work[ret_choice].size > 0:
                    if transitions.state_translate(st,'elig',par) == 1:
                                
                        # satisfying two year rule
                        if transitions.age(t+1,par) >= 62:
                            ret_age[work[ret_choice]] = 0
                                    
                        # not satisfying two year rule
                        elif transitions.age(t+1,par) <= 61:
                            ret_age[work[ret_choice]] = 1

                    else:
                        ret_age[work[ret_choice]] = 2 # no erp

                # f. update a
                a[t,work] = m[t,work] - c[t,work]

            # 5. retired
            if ret.size > 0:    # only if some actually is retired

                # a. optimal consumption choice
                prep = linear_interp.interp_1d_prep(len(ret))
                c_interp = np.nan*np.zeros((len(ret),1))        # initialize
                idx = np.argsort(m[t,ret])                      # indices to sort back later
                m_sort = np.sort(m[t,ret])                      # sort m so interp is faster
                linear_interp.interp_1d_vec_mon(prep,m_sol[t,st,:,0],c_sol[t,st,:,0],m_sort,c_interp[:,0])
                c[t,ret] = c_interp[np.argsort(idx),0]          # sort back

                # b. update retirement choice and probability
                if  t < par.simT-1:     # if not last period
                    d[t+1,ret] = 0      # still retired
                    probs[t+1,ret] = 0  # set retirement prob to 0 if already retired

                # c. update a
                a[t,ret] = m[t,ret] - c[t,ret]   



def euler_error(sim,sol,par):
    """ calculate euler errors
    
        Args:
        
            sim=simulation, sol=solution, par=parameters"""    

    # unpack (to help numba optimize)
    # solution
    c_sol = sol.c[:,:,par.age_dif[0],:,:]
    m_sol = sol.m[:,:,par.age_dif[0],:,:]
    v_sol = sol.v[:,:,par.age_dif[0],:,:]        

    # simulation
    c = sim.c
    m = sim.m
    a = sim.a
    d = sim.d

    # dummies and probabilities
    alive = sim.alive
    ret_age = sim.ret_age                          

    # states
    states = sim.states
    num_st = np.unique(states)

    # misc
    euler = sim.euler
    tol = par.tol


    for t in range(par.simT-2): # cannot calculate for last period
        for st in num_st:

            pi = transitions.survival_look_up(t,st,par)        

            # 1. create masks
            mask_st = np.nonzero(states==st)[0]                                 # mask for state
            work = mask_st[(d[t+1,mask_st]==1) & (alive[t,mask_st]==1)]         # working next period and alive
            ret = mask_st[(d[t+1,mask_st]==0) & (alive[t,mask_st]==1)]          # retired  next period and alive                        
            work_in = work[(tol < c[t,work]) & (c[t,work] < m[t,work] - tol)]   # inner solution for work
            ret_in = ret[(tol < c[t,ret]) & (c[t,ret] < m[t,ret] - tol)]        # inner solution for retired

            # 2. lhs
            euler[t,work_in] = utility.marg_func(c[t,work_in],par)
            euler[t,ret_in] = utility.marg_func(c[t,ret_in],par)

            # 3. rhs
            if ret_in.size > 0:

                # a. retired
                for ra in np.unique(ret_age[ret_in]):   # loop over retirement age (erp status)
                    
                    # next period resources
                    mask_ra = (ret_age[ret_in] == ra) 
                    m_plus = par.R*a[t,ret_in[mask_ra]] + transitions.pension(t+1,st,a[t,ret_in[mask_ra]],ra,par)
                    
                    # interpolation and post-decision
                    c_plus = np.nan*np.zeros((len(ret_in[mask_ra]),1))
                    linear_interp.interp_1d_vec(m_sol[t+1,st,:,0],c_sol[t+1,st,:,0],m_plus,c_plus[:,0])
                    avg_marg_u_plus = utility.marg_func(c_plus[:,0],par)
                    rhs = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)

                    # subtract rhs from lhs
                    euler[t,ret_in[mask_ra]] = euler[t,ret_in[mask_ra]] - rhs

            if work_in.size > 0:

                # b. working
                if transitions.state_translate(st,'male',par) == 1:
                    w = par.xi_men_w
                    xi = par.xi_men
                else:
                    w = par.xi_women_w
                    xi = par.xi_women  

                # prep 
                c_plus = np.nan*np.zeros((len(work_in),2))
                v_plus = np.nan*np.zeros((len(work_in),2))  
                Ra = par.R*a[t,work_in]
                
                # sort
                idx = np.argsort(Ra)    # indices to sort back later
                Ra_sort = np.sort(Ra)   # sort m so interp is faster
                
                # integration and rhs
                avg_marg_u_plus = post_decision.shocks_GH(t,st,Ra_sort,w,xi,c_sol[t+1,st],m_sol[t+1,st],v_sol[t+1,st],c_plus,v_plus,par)[1]
                rhs = par.beta*(par.R*pi*avg_marg_u_plus[np.argsort(idx)] + (1-pi)*par.gamma)   # sort back

                # subtract rhs from lhs
                euler[t,work_in] = euler[t,work_in] - rhs