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


def lifecycle(sim,sol,par,euler=False):
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
    RA = sim.RA                          

    # states
    states = sim.states
    MA = states[:,0]
    NMA = np.unique(MA)
    ST = states[:,1]
    NST = np.unique(ST)

    # random shocks
    choiceP = sim.choiceP
    deadP = sim.deadP
    inc_shock = sim.inc_shock

    # retirement ages
    two_year = transitions.inv_age(par.two_year,par)
    erp_age = transitions.inv_age(par.erp_age,par)        

    # simulation
    # loop over time
    for t in range(par.simT):
        if t > 0:
            alive[t,alive[t-1] == 0] = 0    # still dead

        # loop over gender
        for ma in NMA:        
            mask_ma = np.nonzero(MA==ma)[0]
            pi = transitions.survival_look_up(t,ma,par)
            alive[t,mask_ma[pi < deadP[t,mask_ma]]] = 0    # dead

            # loop over states
            for st in NST:     
                elig = transitions.state_translate(st,'elig',par)
                mask_st = mask_ma[ST[mask_ma]==st]

                # loop over retirement status
                for ra in [0,1,2]:
                    mask = mask_st[RA[mask_st]==ra]                     # mask for ma, st and ra
                    work = mask[(d[t,mask]==1) & (alive[t,mask]==1)]    # working and alive
                    ret = mask[(d[t,mask]==0) & (alive[t,mask]==1)]     # retired and alive            

                    # 1. update m
                    if t > 0:               # m is initialized in 1. period
                        if t < par.Tr-1:    # if not forced retire
                            pre = transitions.labor_pretax(t,ma,st,par)
                            m[t,work] = (par.R*a[t-1,work] + transitions.labor_posttax(t,pre,0,par,inc_shock[t,ma,work]))
                        m[t,ret] = par.R*a[t-1,ret] + transitions.pension(t,ma,st,ra,a[t-1,ret],par)

                    # 2. working
                    if work.size > 0:

                        # a. optimal consumption and value
                        c_interp,v_interp = ConsValue(1,np.array([0,1]),work,t,st,ra,m[t],
                                                      m_sol[t,ma,st],c_sol[t,ma,st],v_sol[t,ma,st],par)

                        # b. retirement prob and optimal choice
                        prob = funs.logsum2(v_interp,par)[1][0] # probs are in 1 and retirement probs are in 0 
                        work_choice = prob < choiceP[t,work]    
                        ret_choice = prob > choiceP[t,work]

                        # c. update consumption (today)
                        c[t,work[work_choice]] = c_interp[1,work_choice]
                        c[t,work[ret_choice]] = c_interp[0,ret_choice]

                        # d. update retirement choice (tomorrow)
                        if t < par.simT-1:

                            # save retirement prob
                            probs[t+1,work] = prob

                            # update retirement choice
                            d[t+1,work[ret_choice]] = 0
                            if t+1 >= par.Tr: # forced to retire
                                d[t+1,work[work_choice]] = 0
                            else:
                                d[t+1,work[work_choice]] = 1

                        # e. update retirement status (everyone are initialized at ra=2)
                        if elig == 1:   # only update if eligible to ERP
            
                            # satisfying two year rule
                            if t+1 >= two_year:
                                RA[work] = 0
                                            
                            # not satisfying two year rule
                            elif t+1 >= erp_age:
                                RA[work] = 1

                        # f. update a
                        a[t,work] = m[t,work] - c[t,work]

                    # 3. retired
                    if ret.size > 0:

                        # a. optimal consumption
                        c_interp = ConsValue(0,np.array([0]),ret,t,st,ra,m[t],
                                             m_sol[t,ma,st],c_sol[t,ma,st],v_sol[t,ma,st],par)[0]                        
                        c[t,ret] = c_interp[0]

                        # b. update retirement choice and probability
                        if t < par.simT-1:      # if not last period
                            d[t+1,ret] = 0      # still retired
                            probs[t+1,ret] = 0  # set retirement prob to 0 if already retired

                        # c. update a
                        a[t,ret] = m[t,ret] - c[t,ret]   

                    # 4. euler erros
                    if euler and t < par.simT-1:
                        euler_error(work,ret,t,ma,st,ra,m,c,a,m_sol,c_sol,v_sol,par,sim)


# def lifecycle_c(sim,sol,par,euler=False):
#     """ Simulate full life-cycle
        
#         Args:

#             sim=simulation, sol=solution, par=parameters"""         

#     # unpack (to help numba optimize)
#     # solution
#     c_sol = sol.c
#     m_sol = sol.m
#     v_sol = sol.v        

#     # simulation
#     c = sim.c
#     m = sim.m
#     a = sim.a
#     d = sim.d

#     # dummies and probabilities
#     alive = sim.alive
#     probs = sim.probs
#     RA = sim.RA                          

#     # states
#     states = sim.States
#     NST = np.unique(states,axis=0)
#     AD = NST[0]
#     ST_H = NST[1]
#     ST_W = NST[2]

#     # random shocks
#     choiceP_h = sim.choiceP_h
#     choiceP_w = sim.choiceP_w    
#     deadP_h = sim.deadP_h
#     deadP_w = sim.deadP_w    
#     inc_shock = sim.inc_shock

#     # retirement ages
#     two_year = transitions.inv_age(par.two_year,par)
#     erp_age = transitions.inv_age(par.erp_age,par)        

#     # simulation
#     # loop over time
#     for t in range(par.simT):
#         if t > 0:
#             for j in [0,1]:
#                 alive[t,alive[t-1,j] == 0,j] = 0    # still dead

#         for ad in AD:
#             mask_ad = np.nonzero(AD==ad)[0]
#             pi_h,pi_w = transitions.survival_look_up_c(t,ad,par)
#             alive[t,mask_ad[pi_h < deadP_h[t,mask_ad]],0]       # husband
#             alive[t,mask_ad[pi_w < deadP_w[t+ad,mask_ad]],1]    # wife

#             for st_h in ST_h:
#                 elig_h = transitions.state_translate(st_h,'elig',par)
#                 mask_stH = mask_ad[]

#         # loop over gender
#         for ma in NMA:        
#             mask_ma = np.nonzero(MA==ma)[0]
#             pi = transitions.survival_look_up(t,ma,par)
#             alive[t,mask_ma[pi < deadP[t,mask_ma]]] = 0    # dead

#             # loop over states
#             for st in NST:     
#                 elig = transitions.state_translate(st,'elig',par)
#                 mask_st = mask_ma[ST[mask_ma]==st]

#                 # loop over retirement status
#                 for ra in [0,1,2]:
#                     mask = mask_st[RA[mask_st]==ra]                     # mask for ma, st and ra
#                     work = mask[(d[t,mask]==1) & (alive[t,mask]==1)]    # working and alive
#                     ret = mask[(d[t,mask]==0) & (alive[t,mask]==1)]     # retired and alive            

#                     # 1. update m
#                     if t > 0:               # m is initialized in 1. period
#                         if t < par.Tr-1:    # if not forced retire
#                             pre = transitions.labor_pretax(t,ma,st,par)
#                             m[t,work] = (par.R*a[t-1,work] + transitions.labor_posttax(t,pre,0,par,inc_shock[t,ma,work]))
#                         m[t,ret] = par.R*a[t-1,ret] + transitions.pension(t,ma,st,ra,a[t-1,ret],par)

#                     # 2. working
#                     if work.size > 0:

#                         # a. optimal consumption and value
#                         c_interp,v_interp = ConsValue(1,np.array([0,1]),work,t,st,ra,m[t],
#                                                       m_sol[t,ma,st],c_sol[t,ma,st],v_sol[t,ma,st],par)

#                         # b. retirement prob and optimal choice
#                         prob = funs.logsum2(v_interp,par)[1][0] # probs are in 1 and retirement probs are in 0 
#                         work_choice = prob < choiceP[t,work]    
#                         ret_choice = prob > choiceP[t,work]

#                         # c. update consumption (today)
#                         c[t,work[work_choice]] = c_interp[1,work_choice]
#                         c[t,work[ret_choice]] = c_interp[0,ret_choice]

#                         # d. update retirement choice (tomorrow)
#                         if t < par.simT-1:

#                             # save retirement prob
#                             probs[t+1,work] = prob

#                             # update retirement choice
#                             d[t+1,work[ret_choice]] = 0
#                             if t+1 >= par.Tr: # forced to retire
#                                 d[t+1,work[work_choice]] = 0
#                             else:
#                                 d[t+1,work[work_choice]] = 1

#                         # e. update retirement status (everyone are initialized at ra=2)
#                         if elig == 1:   # only update if eligible to ERP
            
#                             # satisfying two year rule
#                             if t+1 >= two_year:
#                                 RA[work] = 0
                                            
#                             # not satisfying two year rule
#                             elif t+1 >= erp_age:
#                                 RA[work] = 1

#                         # f. update a
#                         a[t,work] = m[t,work] - c[t,work]

#                     # 3. retired
#                     if ret.size > 0:

#                         # a. optimal consumption
#                         c_interp = ConsValue(0,np.array([0]),ret,t,st,ra,m[t],
#                                              m_sol[t,ma,st],c_sol[t,ma,st],v_sol[t,ma,st],par)[0]                        
#                         c[t,ret] = c_interp[0]

#                         # b. update retirement choice and probability
#                         if t < par.simT-1:      # if not last period
#                             d[t+1,ret] = 0      # still retired
#                             probs[t+1,ret] = 0  # set retirement prob to 0 if already retired

#                         # c. update a
#                         a[t,ret] = m[t,ret] - c[t,ret]   

#                     # 4. euler erros
#                     if euler and t < par.simT-1:
#                         euler_error(work,ret,t,ma,st,ra,m,c,a,m_sol,c_sol,v_sol,par,sim)


@njit(parallel=True)
def ConsValue(dummy,D,mask,t,st,ra,m,m_sol,c_sol,v_sol,par):

    # a. initialize
    prep = linear_interp.interp_1d_prep(len(mask))
    c_interp = np.zeros((len(D),len(mask)))
    v_interp = np.zeros((len(D),len(mask)))
    ra_look = transitions.ra_look_up(t,st,ra,dummy,par)

    # b. sort m (so interp is faster)
    idx = np.argsort(np.argsort(m[mask])) # index to unsort 
    m_sort = np.sort(m[mask])

    # c. interpolate and sort back
    if dummy == 1:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[ra_look,d],c_sol[ra_look,d],m_sort,c_interp[d])        
            linear_interp.interp_1d_vec_mon(prep,m_sol[ra_look,d],v_sol[ra_look,d],m_sort,v_interp[d])         

        # sort back
        c_interp = c_interp[:,idx]
        v_interp = v_interp[:,idx]

    elif dummy == 0:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[ra_look,d],c_sol[ra_look,d],m_sort,c_interp[d])            

        # sort back
        c_interp = c_interp[:,idx]

    # d. return
    return c_interp,v_interp


def euler_error(work,ret,t,ma,st,ra,m,c,a,m_sol,c_sol,v_sol,par,sim):
    """ calculate euler errors
    
        Args:
        
            sim=simulation, sol=solution, par=parameters"""    

    # unpack
    m_sol = m_sol[t+1,ma,st]
    c_sol = c_sol[t+1,ma,st]
    v_sol = v_sol[t+1,ma,st]
    tol = par.tol
    euler = sim.euler
    pi_plus = transitions.survival_look_up(t+1,ma,par)    

    # 1. mask
    work_in = work[(tol < c[t,work]) & (c[t,work] < m[t,work] - tol)]   # inner solution for work
    ret_in = ret[(tol < c[t,ret]) & (c[t,ret] < m[t,ret] - tol)]        # inner solution for retired

    # 2. lhs
    euler[t,work_in] = utility.marg_func(c[t,work_in],par)
    euler[t,ret_in] = utility.marg_func(c[t,ret_in],par)

    # 3. rhs
    if ret_in.size > 0:

        # a. next period income and retirement age
        m_plus = par.R*a[t,ret_in] + transitions.pension(t+1,ma,st,ra,a[t,ret_in],par)
        ra_plus = transitions.ra_look_up(t+1,st,ra,0,par)

        # b. interpolation
        c_plus = np.nan*np.zeros((1,len(ret_in)))
        linear_interp.interp_1d_vec(m_sol[ra_plus,0],c_sol[ra_plus,0],m_plus,c_plus[0])
        
        # c. post-decision (rhs)
        avg_marg_u_plus = utility.marg_func(c_plus[0],par)
        rhs = par.beta*(par.R*pi_plus*avg_marg_u_plus + (1-pi_plus)*par.gamma)

        # d. subtract rhs from lhs
        euler[t,ret_in] = euler[t,ret_in] - rhs

    if work_in.size > 0:

        # a. next period income, retirement age and choice set
        Ra = par.R*a[t,work_in]
        inc = transitions.labor_look_up(t+1,ma,st,par)
        ra_plus = transitions.ra_look_up(t+1,st,ra,1,par)        
        d_plus = np.array([0,1])

        # b. prep for integration
        c_plus = np.nan*np.zeros((2,len(work_in)))
        v_plus = np.nan*np.zeros((2,len(work_in)))  
        if ma == 1:
            w = par.xi_men_w
        else:
            w = par.xi_women_w 

        # c. sort
        idx = np.argsort(np.argsort(Ra))    # indices to sort back later
        Ra_sort = np.sort(Ra)               # sort Ra so interp is faster
                    
        # d. integration and rhs
        avg_marg_u_plus = post_decision.shocks_GH(t,Ra_sort,inc,w,c_sol[ra_plus],m_sol[ra_plus],v_sol[ra_plus],c_plus,v_plus,par,d_plus)[1]
        rhs = par.beta*(par.R*pi_plus*avg_marg_u_plus[idx] + (1-pi_plus)*par.gamma)   # sort back using idx

        # e. subtract rhs from lhs
        euler[t,work_in] = euler[t,work_in] - rhs



def lifecycle_c(sim,sol,par,euler=False):
    """ Simulate full life-cycle
        
        Args:

            sim=simulation, sol=solution, par=parameters"""         

    # unpack (to help numba optimize)
    # solution
    c_sol = sol.c
    m_sol = sol.m
    v_sol = sol.v        

    # simulation
    c = sim.c
    m = sim.m
    a = sim.a
    d = sim.d

    # dummies and probabilities
    alive = sim.alive
    probs = sim.probs
    RA = sim.RA                          

    # states
    MA = sim.MA
    NMA = np.unique(MA)
    ST = sim.ST
    NST = np.unique(ST)

    # random shocks
    choiceP = sim.choiceP
    deadP = sim.deadP
    inc_shock = sim.inc_shock

    # retirement ages
    two_year = transitions.inv_age(par.two_year,par)
    erp_age = transitions.inv_age(par.erp_age,par)        

    # simulation
    # loop over time
    for t in range(par.simT):
        if t > 0:
            alive[t,alive[t-1] == 0] = 0    # still dead

        # loop over gender
        for ma in NMA:        
            mask_ma = np.nonzero(MA==ma)[0]
            pi = transitions.survival_look_up(t,ma,par)
            alive[t,mask_ma[pi < deadP[t,mask_ma]]] = 0    # dead

            # loop over states
            for st in NST:     
                elig = transitions.state_translate(st,'elig',par)
                mask_st = mask_ma[ST[mask_ma]==st]

                # loop over retirement status
                for ra in [0,1,2]:
                    mask = mask_st[RA[mask_st]==ra]                     # mask for ma, st and ra
                    work = mask[(d[t,mask]==1) & (alive[t,mask]==1)]    # working and alive
                    ret = mask[(d[t,mask]==0) & (alive[t,mask]==1)]     # retired and alive            

                    # 1. update m
                    if t > 0:               # m is initialized in 1. period
                        if t < par.Tr-1:    # if not forced retire
                            pre = transitions.labor_pretax(t,ma,st,par)
                            m[t,work] = (par.R*a[t-1,work] + transitions.labor_posttax(t,pre,0,par,inc_shock[t,ma,work]))
                        m[t,ret] = par.R*a[t-1,ret] + transitions.pension(t,ma,st,ra,a[t-1,ret],par)

                    # 2. working
                    if work.size > 0:

                        # a. optimal consumption and value
                        c_interp,v_interp = ConsValue(1,np.array([0,1]),work,t,st,ra,m[t],
                                                      m_sol[t,ma,st],c_sol[t,ma,st],v_sol[t,ma,st],par)

                        # b. retirement prob and optimal choice
                        prob = funs.logsum2(v_interp,par)[1][0] # probs are in 1 and retirement probs are in 0 
                        work_choice = prob < choiceP[t,work]    
                        ret_choice = prob > choiceP[t,work]

                        # c. update consumption (today)
                        c[t,work[work_choice]] = c_interp[1,work_choice]
                        c[t,work[ret_choice]] = c_interp[0,ret_choice]

                        # d. update retirement choice (tomorrow)
                        if t < par.simT-1:

                            # save retirement prob
                            probs[t+1,work] = prob

                            # update retirement choice
                            d[t+1,work[ret_choice]] = 0
                            if t+1 >= par.Tr: # forced to retire
                                d[t+1,work[work_choice]] = 0
                            else:
                                d[t+1,work[work_choice]] = 1

                        # e. update retirement status (everyone are initialized at ra=2)
                        if elig == 1:   # only update if eligible to ERP
            
                            # satisfying two year rule
                            if t+1 >= two_year:
                                RA[work] = 0
                                            
                            # not satisfying two year rule
                            elif t+1 >= erp_age:
                                RA[work] = 1

                        # f. update a
                        a[t,work] = m[t,work] - c[t,work]

                    # 3. retired
                    if ret.size > 0:

                        # a. optimal consumption
                        c_interp = ConsValue(0,np.array([0]),ret,t,st,ra,m[t],
                                             m_sol[t,ma,st],c_sol[t,ma,st],v_sol[t,ma,st],par)[0]                        
                        c[t,ret] = c_interp[0]

                        # b. update retirement choice and probability
                        if t < par.simT-1:      # if not last period
                            d[t+1,ret] = 0      # still retired
                            probs[t+1,ret] = 0  # set retirement prob to 0 if already retired

                        # c. update a
                        a[t,ret] = m[t,ret] - c[t,ret]   

                    # 4. euler erros
                    if euler and t < par.simT-1:
                        euler_error(work,ret,t,ma,st,ra,m,c,a,m_sol,c_sol,v_sol,par,sim)