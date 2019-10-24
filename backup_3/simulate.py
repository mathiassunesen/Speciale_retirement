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
    # simulation
    c = sim.c
    m = sim.m
    a = sim.a
    d = sim.d

    # dummies and probabilities
    alive = sim.alive[:,:,0]             

    # states
    states = sim.states
    MA = states[:,0]
    NMA = np.unique(MA)
    ST = states[:,1]
    NST = np.unique(ST)
    RA = sim.RA[:,0]

    # random shocks
    deadP = sim.deadP[:,:,0]

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

                    # 1. working
                    if work.size > 0:

                        # update m (initialized in 1. period)
                        if t > 0:
                            update_m(t,ma,st,ra,1,sim,par,work)                        

                        # optimal choices
                        c_interp,v_interp = ConsValue(t,ma,st,ra,1,sol,par,sim,work)                        
                        optimal_choices(t,1,c_interp,v_interp,sim,par,work)

                        # update retirement age
                        update_ra(t,elig,two_year,erp_age,sim,work)

                        # update a
                        a[t,work] = m[t,work] - c[t,work]

                    # 2. retired
                    if ret.size > 0:

                        # update m (initialized in 1. period)
                        if t > 0:
                            update_m(t,ma,st,ra,0,sim,par,ret)                         

                        # optimal choices
                        c_interp,v_interp = ConsValue(t,ma,st,ra,0,sol,par,sim,ret)                        
                        optimal_choices(t,0,c_interp,v_interp,sim,par,ret)                        

                        # update a
                        a[t,ret] = m[t,ret] - c[t,ret]   

                    # 3. euler erros
                    if euler and t < par.simT-1:
                        euler_error(t,ma,st,ra,sol,par,sim,work,ret)

@njit(parallel=True)
def update_m(t,ma,st,ra,ds,sim,par,mask):

    # unpack
    m = sim.m[t]
    a = sim.a[t-1]

    if ds == 1:
        inc_shock = sim.inc_shock[t,:,ma]
        pre = transitions.labor_pretax(t,ma,st,par)
        m[mask] = par.R*a[mask] + transitions.labor_posttax(t,pre,0,par,inc_shock[mask])

    elif ds == 0:
        m[mask] = par.R*a[mask] + transitions.pension(t,ma,st,ra,a[mask],par)        

@njit(parallel=True)
def ConsValue(t,ma,st,ra,ds,sol,par,sim,mask):

    # a. unpack
    D = transitions.d_plus(t-1,ds,par)
    ra_look = transitions.ra_look_up(t,st,ra,ds,par)
    c_sol = sol.c[t,0,ma,st,ra_look]
    m_sol = sol.m[t,0,ma,st,ra_look]
    v_sol = sol.v[t,0,ma,st,ra_look]    
    m = sim.m[t]        

    # a. initialize
    prep = linear_interp.interp_1d_prep(len(mask))
    c_interp = np.zeros((len(D),len(mask)))
    v_interp = np.zeros((len(D),len(mask)))

    # b. sort m (so interp is faster)
    idx = np.argsort(np.argsort(m[mask])) # index to unsort 
    m_sort = np.sort(m[mask])

    # c. interpolate and sort back
    if ds == 1:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[d],c_sol[d],m_sort,c_interp[d])        
            linear_interp.interp_1d_vec_mon(prep,m_sol[d],v_sol[d],m_sort,v_interp[d])         

        # sort back
        c_interp = c_interp[:,idx]
        v_interp = v_interp[:,idx]

    elif ds == 0:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[d],c_sol[d],m_sort,c_interp[d])            

        # sort back
        c_interp = c_interp[:,idx]

    # d. return
    return c_interp,v_interp

@njit(parallel=True)
def optimal_choices(t,ds,c_interp,v_interp,sim,par,mask):

    # unpack
    c = sim.c[t]
    choiceP = sim.choiceP[t,:,0]
    if t < par.simT-1:
        d = sim.d[t+1] 
        probs = sim.probs[t+1,:,0]       

    # working
    if ds == 1:

        # a. retirement prob and optimal labor choice
        prob = funs.logsum2(v_interp,par)[1][0] # probs are in 1 and retirement probs are in 0 
        work_choice = prob < choiceP[mask]    
        ret_choice = prob > choiceP[mask]

        # b. optimal consumption (today)
        mask_w = mask[work_choice]
        mask_r = mask[ret_choice]
        c[mask_w] = c_interp[1,work_choice]
        c[mask_r] = c_interp[0,ret_choice]

        # c. optimal retirement choice (tomorrow)
        if t < par.simT-1:

            # save retirement prob
            probs[mask] = prob

            # update retirement choice
            d[mask_r] = 0
            if t+1 >= par.Tr: # forced to retire
                d[mask_w] = 0
            else:
                d[mask_w] = 1   

    # retired
    elif ds == 0:
        
        # a. optimal consumption (today)
        c[mask] = c_interp[0]
        
        # b. retirement choice is fixed
        if t < par.simT-1:  # if not last period
            d[mask] = 0     # still retired
            probs[mask] = 0 # set retirement prob to 0 if already retired    

@njit(parallel=True)
def update_ra(t,elig,two_year,erp_age,sim,mask):

    # unpack
    RA = sim.RA[:,0] 

    # update retirement status (everyone are initialized at ra=2)
    if elig == 1:   # only update if eligible to ERP
            
        # satisfying two year rule
        if t+1 >= two_year:
            RA[mask] = 0
                                            
        # not satisfying two year rule
        elif t+1 >= erp_age:
            RA[mask] = 1


@njit(parallel=True)
def euler_error(t,ma,st,ra,sol,par,sim,work,ret):
    """ calculate euler errors
    
        Args:
        
            sim=simulation, sol=solution, par=parameters"""    

    # unpack
    m_sol = sol.m[t+1,0,ma,st]
    c_sol = sol.c[t+1,0,ma,st]
    v_sol = sol.v[t+1,0,ma,st]
    c = sim.c[t]
    m = sim.m[t]
    a = sim.a[t]
    tol = par.tol
    euler = sim.euler[t]
    pi_plus = transitions.survival_look_up(t+1,ma,par)    

    # 1. mask
    work_in = work[(tol < c[work]) & (c[work] < m[work] - tol)]   # inner solution for work
    ret_in = ret[(tol < c[ret]) & (c[ret] < m[ret] - tol)]        # inner solution for retired

    # 2. lhs
    euler[work_in] = utility.marg_func(c[work_in],par)
    euler[ret_in] = utility.marg_func(c[ret_in],par)

    # 3. rhs
    if ret_in.size > 0:

        # a. next period income and retirement age
        m_plus = par.R*a[ret_in] + transitions.pension(t+1,ma,st,ra,a[ret_in],par)
        ra_plus = transitions.ra_look_up(t+1,st,ra,0,par)

        # b. interpolation
        c_plus = np.nan*np.zeros((1,len(ret_in)))
        linear_interp.interp_1d_vec(m_sol[ra_plus,0],c_sol[ra_plus,0],m_plus,c_plus[0])
        
        # c. post-decision (rhs)
        avg_marg_u_plus = utility.marg_func(c_plus[0],par)
        rhs = par.beta*(par.R*pi_plus*avg_marg_u_plus + (1-pi_plus)*par.gamma)

        # d. subtract rhs from lhs
        euler[ret_in] = euler[ret_in] - rhs

    if work_in.size > 0:

        # a. next period income, retirement age and choice set
        Ra = par.R*a[work_in]
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
        euler[work_in] = euler[work_in] - rhs



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
    ALIVE_H = sim.alive[:,:,0]
    ALIVE_W = sim.alive[:,:,1]
    probs_h = sim.probs[:,:,0]
    probs_w = sim.probs[:,:,1]
    RA_H = sim.RA[:,0]
    RA_W = sim.RA[:,1]                          

    # states
    states = sim.States
    NST = np.unique(states,axis=0)
    AD = NST[0]
    ST_H = NST[1]
    ST_W = NST[2]

    # random shocks
    choiceP_h = sim.choiceP[:,:,0]
    choiceP_w = sim.choiceP[:,:,1]    
    deadP_h = sim.deadP[:,:,0]
    deadP_w = sim.deadP[:,:,1]    
    inc_shock_h = sim.inc_shock[:,:,1]
    inc_shock_w = sim.inc_shock[:,:,0]
    inc_shock_c = sim.inc_shock[:,:,2]    

    # retirement ages
    two_year = transitions.inv_age(par.two_year,par)
    erp_age = transitions.inv_age(par.erp_age,par)        

    # simulation
    # loop over time
    for t in range(par.simT):
        if t > 0:
            alive_h[t,alive_h[t-1] == 0] = 0    # still dead
            alive_w[t,alive_w[t-1] == 0] = 0
            
        for ad in AD:   # age difference
            mask_ad = np.nonzero(AD==ad)[0]
            pi_h,pi_w = transitions.survival_look_up_c(t,ad,par)
            alive_h[t,mask_ad[pi_h < deadP_h[t,mask_ad]]] = 0               
            alive_w[t,mask_ad[pi_w < deadP_w[t,mask_ad]]] = 0   

            for st_h in ST_H:   # states of husband
                elig_h = transitions.state_translate(st_h,'elig',par)
                mask_stH = mask_ad[ST_H[mask_ad]==st_h]

                for st_w in ST_W:   # states of wife
                    elig_w = transitions.state_translate(st_w,'elig',par)
                    mask_stW = mask_stH[ST_W[mask_stH]==st_w]

                    for ra_h in [0,1,2]:    # retirement age of husband
                        mask_raH = mask_stW[RA_H[mask_stW]==ra_h]

                        for ra_w in [0,1,2]:    # retirement age of wife
                            mask_raW = mask_raH[RA_W[mask_raH]==ra_w]




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


@njit(parallel=True)
def ConsValue_c(t,ad,st_h,st_w,ra_h,ra_w,d_h,d_w,sol,par,single_sol,mask,m,alive_h,alive_w):

    # a. unpack solution
    if alive_h == 1 and alive_w == 1:
        D = transitions.d_plus_c(t-1,ad,d_h,d_w,par)
        ad_min = par.ad_min
        ad_idx = ad+ad_min        
        ra_look_h = transitions.ra_look_up(t,st_h,ra_h,d_h,par)
        ra_look_w = transitions.ra_look_up(t+ad,st_w,ra_w,d_w,par)    
        c_sol = sol.c[t,ad_idx,st_h,st_w,ra_look_h,ra_look_w]
        m_sol = sol.m[t,ad_idx,st_h,st_w,ra_look_h,ra_look_w]
        v_sol = sol.v[t,ad_idx,st_h,st_w,ra_look_h,ra_look_w]

    elif alive_h == 1 and alive_w == 0:
        D = transitions.d_plus(t-1,d_h,par)
        ra_look_h = transitions.ra_look_up(t,st_h,ra_h,d_h,par)
        c_sol = single_sol.c[t,0,st_h,1,ra_look_h]
        m_sol = single_sol.m[t,0,st_h,1,ra_look_h]
        v_sol = single_sol.v[t,0,st_h,1,ra_look_h]        

    elif alive_h == 0 and alive_w == 1:
        D = transitions.d_plus(t-1+ad,d_w,par)
        ra_look_w = transitions.ra_look_up(t,st_w,ra_w,d_w,par)
        c_sol = single_sol.c[t,0,st_w,0,ra_look_w]
        m_sol = single_sol.m[t,0,st_w,0,ra_look_w]
        v_sol = single_sol.v[t,0,st_w,0,ra_look_w]   

    # b. initialize     
    ND = len(D)
    NM = len(mask)    
    prep = linear_interp.interp_1d_prep(NM)
    c_interp = np.zeros((ND,NM))
    v_interp = np.zeros((ND,NM))

    # c. sort m (so interp is faster)
    idx = np.argsort(np.argsort(m[mask])) # index to unsort 
    m_sort = np.sort(m[mask])

    # c. interpolate and sort back
    for d in D:
        linear_interp.interp_1d_vec_mon(prep,m_sol[d],c_sol[d],m_sort,c_interp[d])        
        linear_interp.interp_1d_vec_mon(prep,m_sol[d],v_sol[d],m_sort,v_interp[d])         
    c_interp = c_interp[:,idx]
    v_interp = v_interp[:,idx]

    # d. return
    return c_interp,v_interp
