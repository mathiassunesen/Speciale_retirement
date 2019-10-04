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
    c_sol = sol.c[:,0]    # ad=0
    m_sol = sol.m[:,0]
    v_sol = sol.v[:,0]        

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
    male = sim.male
    num_male = np.unique(male)
    states = sim.states
    num_st = np.unique(states)

    # random shocks
    unif = sim.unif
    deadP = sim.deadP
    inc_shock = sim.inc_shock

    # retirement ages
    oap_age = transitions.inv_age(par.oap_age,par)
    two_year = transitions.inv_age(par.two_year,par)
    erp_age = transitions.inv_age(par.erp_age,par)        

    # simulation
    for t in range(par.simT):   # loop over time

        if t > 0:
            alive[t,alive[t-1] == 0] = 0    # still dead

        for ma in num_male:         # loop over gender
            for st in num_st:       # loop over states
                for ra in [0,1,2]:  # loop over retirement status
            
                    # 1. update alive
                    mask = np.nonzero(states==st)[0]
                    mask = mask[male[mask]==ma]            
                    mask = mask[ret_age[mask]==ra]
                    pi = transitions.survival_look_up(t,ma,par)
                    alive[t,mask[pi < deadP[t,mask]]] = 0     # dead

                    # 2. create masks
                    work = mask[(d[t,mask]==1) & (alive[t,mask]==1)]   # mask for state, work and alive
                    ret = mask[(d[t,mask]==0) & (alive[t,mask]==1)]    # mask for state, retired and alive            

                    # 3. update m
                    if t > 0:               # m is initialized in 1. period
                        if t < par.Tr-1:    # if not forced retire
                            pre = transitions.labor_pretax(t,ma,st,par)
                            m[t,work] = (par.R*a[t-1,work] + transitions.labor_posttax(t,pre,0,par,inc_shock[t,ma,work]))
                        m[t,ret] = par.R*a[t-1,ret] + transitions.pension(t,ma,st,ra,a[t-1,ret],par)

                    # 4. working
                    if t < par.Tr-1 and work.size > 0:  # not forced to retire and working
            
                        # a. interpolation
                        prep = linear_interp.interp_1d_prep(len(work))  
                        c_interp = np.nan*np.zeros((2,len(work)))
                        v_interp = np.nan*np.zeros((2,len(work))) 
                        index = np.argsort(m[t,work]) # indices to sort back later
                        m_sort = np.sort(m[t,work]) # sort m so interp is faster
                        for idx in range(2):
                            # if idx == 0:      # dette giver gode forbrugsfunktioner men dårlige ssh
                            #     ra_look = transitions.ra_look_up(t,st,ra,idx,par)
                            # elif idx == 1:
                            #     ra_look = transitions.ra_look_up(t,st,ra,idx,par)
                            ra_look = transitions.ra_look_up(t,st,ra,1,par) # dette giver dårlige forbrugsfunktioner men dårlige ssh
                            linear_interp.interp_1d_vec_mon(prep,m_sol[t,ma,st,ra_look,idx,:],c_sol[t,ma,st,ra_look,idx,:],m_sort,c_interp[idx,:])
                            linear_interp.interp_1d_vec_mon_rep(prep,m_sol[t,ma,st,ra_look,idx,:],v_sol[t,ma,st,ra_look,idx,:],m_sort,v_interp[idx,:])

                        # sort back
                        c_interp = c_interp[:,np.argsort(index)]
                        v_interp = v_interp[:,np.argsort(index)]

                        # b. retirement probabilities
                        prob = funs.logsum2(v_interp,par)[1]
                        work_choice = prob[0,:] < unif[t,work]    
                        ret_choice = prob[0,:] > unif[t,work]

                        # c. optimal choices for today (consumption)
                        c[t,work[work_choice]] = c_interp[1,work_choice]
                        c[t,work[ret_choice]] = c_interp[0,ret_choice]

                        # d. optimal choices for tomorrow (retirement)
                        if t < par.simT-1:
                            probs[t+1,work] = prob[0,:] # save retirement probability
                            d[t+1,work[ret_choice]] = 0
                            if t+1 >= par.Tr: # forced to retire
                                d[t+1,work[work_choice]] = 0
                            else:
                                d[t+1,work[work_choice]] = 1

                        # e. update retirement age (erp status)
                        if work.size > 0:
                            if transitions.state_translate(st,'elig',par) == 1:
                                        
                                # satisfying two year rule
                                if t+1 >= two_year:
                                    ret_age[work] = 0
                                            
                                # not satisfying two year rule
                                elif t+1 >= erp_age:
                                    ret_age[work] = 1

                        # f. update a
                        a[t,work] = m[t,work] - c[t,work]

                    # 5. retired
                    if ret.size > 0:    # only if some actually is retired

                        # a. optimal consumption choice
                        prep = linear_interp.interp_1d_prep(len(ret))
                        c_interp = np.nan*np.zeros((1,len(ret)))        # initialize
                        index = np.argsort(m[t,ret])                      # indices to sort back later
                        m_sort = np.sort(m[t,ret])                      # sort m so interp is faster
                        ra_look = transitions.ra_look_up(t,st,ra,0,par)
                        linear_interp.interp_1d_vec_mon(prep,m_sol[t,ma,st,ra_look,0,:],c_sol[t,ma,st,ra_look,0,:],m_sort,c_interp[0,:])
                        c[t,ret] = c_interp[0,np.argsort(index)]          # sort back

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
    c_sol = sol.c[:,0,:,:,0,:]    # ad=0 and erp=0
    m_sol = sol.m[:,0,:,:,0,:]
    v_sol = sol.v[:,0,:,:,0,:]          

    # simulation
    c = sim.c
    m = sim.m
    a = sim.a
    d = sim.d

    # dummies and probabilities
    alive = sim.alive
    ret_age = sim.ret_age                          

    # states
    male = sim.male
    num_male = np.unique(male)
    states = sim.states
    num_st = np.unique(states)

    # misc
    euler = sim.euler
    tol = par.tol

    for t in range(par.simT-2): # cannot calculate for last period
        for ma in num_male: 
            for st in num_st:

                pi = transitions.survival_look_up(t,ma,par)        

                # 1. create masks
                mask_st = np.nonzero(states==st)[0]
                mask_st = mask_st[male[mask_st]==ma]                                # mask of gender and states                
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
                        m_plus = par.R*a[t,ret_in[mask_ra]] + transitions.pension(t+1,ma,st,ra,a[t,ret_in[mask_ra]],par)
                        
                        # interpolation and post-decision
                        c_plus = np.nan*np.zeros((1,len(ret_in[mask_ra])))
                        linear_interp.interp_1d_vec(m_sol[t+1,ma,st,0,:],c_sol[t+1,ma,st,0,:],m_plus,c_plus[0,:])
                        avg_marg_u_plus = utility.marg_func(c_plus[0,:],par)
                        rhs = par.beta*(par.R*pi*avg_marg_u_plus + (1-pi)*par.gamma)

                        # subtract rhs from lhs
                        euler[t,ret_in[mask_ra]] = euler[t,ret_in[mask_ra]] - rhs

                if work_in.size > 0:

                    # b. working
                    if ma == 1:
                        w = par.xi_men_w
                    else:
                        w = par.xi_women_w 

                    # prep 
                    c_plus = np.nan*np.zeros((2,len(work_in)))
                    v_plus = np.nan*np.zeros((2,len(work_in)))  
                    Ra = par.R*a[t,work_in]
                    inc = transitions.labor_look_up(t+1,ma,st,par)
                    d_lst = np.array([0,1])
                    
                    # sort
                    idx = np.argsort(Ra)    # indices to sort back later
                    Ra_sort = np.sort(Ra)   # sort m so interp is faster
                    
                    # integration and rhs
                    avg_marg_u_plus = post_decision.shocks_GH(t,Ra_sort,inc,w,c_sol[t+1,ma,st],m_sol[t+1,ma,st],v_sol[t+1,ma,st],c_plus,v_plus,par,d_lst)[1]
                    rhs = par.beta*(par.R*pi*avg_marg_u_plus[np.argsort(idx)] + (1-pi)*par.gamma)   # sort back

                    # subtract rhs from lhs
                    euler[t,work_in] = euler[t,work_in] - rhs