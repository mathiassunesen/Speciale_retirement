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

##################################
####      help functions     #####
##################################
@njit(parallel=True)
def fill_arr(y_arr,idx,x_arr):
    """ fill out the array y with the array x at the indices idx"""    
    for i in range(idx.size):
        y_arr[idx[i]] = x_arr[i]

@njit(parallel=True)
def fill_number(y_arr,idx,x):
    """ fill out the array y with the number x at the indices idx"""        
    for i in range(idx.size):
        y_arr[idx[i]] = x        

##################################
####         Singles         #####
##################################
@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ wrapper for simulating single model """

    # unpack
    RA = sim.RA
    alive = sim.alive
    d = sim.d

    # states which are fixed (parallel outside loop)
    MA = sim.states[:,0]
    ST = sim.states[:,1]
    it = par.iterator
    for i in prange(len(it)):
        ma = it[i,0]
        st = it[i,1]
        idx_st = np.nonzero((MA==ma) & (ST==st))[0]
        elig = transitions.state_translate(st,'elig',par)

        # loop over time and retirement status
        for t in range(par.simT):
            for ra in np.array([0,1,2]):

                # indices
                idx_alive = idx_st[(RA[idx_st]==ra) & (alive[idx_st,t]==1)]
                
                # simulate
                for ds in [0,1]:
                    idx = idx_alive[d[idx_alive,t]==ds]
                    simulate_single(t,ma,st,elig,ra,sim.c,sim.m,sim.a,sim.d,sim.probs,sim.RA,sim.GovS,sol,par,sim,idx,ds)

                    # euler errors
                    if sim.accuracy and t < par.simT-1:
                        euler_error(t,ma,st,ra,sim.euler,sol,par,sim,idx,ds)                   

@njit(parallel=True)
def simulate_single(t,ma,st,elig,ra,c,m,a,d,probs,RA,GovS,sol,par,sim,idx,ds,ad=0,ad_min=0):
    """ simulate single model (ad and ad_min are only so the couple model can look up in this function) """

    if idx.size > 0:

        # update m and optimal choices
        if t > 0:   # m is initialized in 1. period
            update_m(t,ma,st,ra,ds,m,a,sim,par,idx,GovS,ad,ad_min)
        c_interp,v_interp = ConsValue(t,ma,st,ra,ds,m,sol,par,idx,ad,ad_min)
        optimal_choices(t,ds,c,d,probs,c_interp,v_interp,sim,par,idx,ad,ad_min)

        # only update retirement status if still working
        if ds == 1:
            update_ra(t,elig,RA,par,idx)
        
        # update a
        fill_arr(a[:,t],idx,m[idx,t]-c[idx,t])        

@njit(parallel=True)
def update_ra(t,elig,RA,par,idx):
    """ update retirement status (ra) """

    if elig == 1:   # only update if eligible to ERP
        if t+1 >= par.T_two_year:
            fill_number(RA[:],idx,0)
        elif t+1 >= par.T_erp:
            fill_number(RA[:],idx,1)               

@njit(parallel=True)
def update_m(t,ma,st,ra,ds,m,a,sim,par,idx,GovS,ad=0,ad_min=0):
    """ update m for singles """

    # unpack
    a_idx = sim.a[idx,t-1]

    # working
    if ds == 1:
        inc = sim.labor_post[idx,t+ad+ad_min,ma]

    # retired
    elif ds == 0:
        shocks = np.ones(len(idx))

        # oap
        if t >= par.T_oap:
            pre = transitions.oap_pretax(t,par,i=0)[0]*shocks
            inc = transitions.posttax(t,par,ds,inc=0*shocks,pens=pre,spouse_inc=np.inf*shocks)  # set spouse inc to inf, so when we look up for couples (who are now singles) they can't share deduction

        # erp
        elif par.T_erp <= t < par.T_oap:
            pre = transitions.erp_pretax(t,ma,st,ra,par)[0]*shocks
            inc = transitions.posttax(t,par,ds,inc=0*shocks,pens=pre,spouse_inc=np.inf*shocks)

        else:
            pre = np.zeros(len(idx))
            inc = pre

    # update m
    fill_arr(m[:,t],idx,par.R*a_idx[:] + inc[:])

    # government surplus
    if sim.tax:
        if ds == 1:
            pre = sim.labor_pre[idx,t,ma]        
        fill_arr(GovS[:,t],idx, ds*pre[:] - inc[:])     # working (ds=1): tax is pre-inc, 
                                                        # retired (ds=0): tax is -inc                    

@njit(parallel=True)
def ConsValue(t,ma,st,ra,ds,m,sol,par,idx,ad=0,ad_min=0):
    """ interpolate consumption and value (ad and ad_min are only so the couple model can look up in this function) """

    # a. unpack solution
    ad_idx = ad+ad_min    
    D = transitions.d_plus(t+ad-1,ds,par)   # t-1 so we get choice set today
    ra_look = transitions.ra_look_up(t+ad,st,ra,ds,par)
    c_sol = sol.c[t+ad_idx,ma,st,ra_look]
    m_sol = sol.m[:]
    v_sol = sol.v[t+ad_idx,ma,st,ra_look]    

    # a. initialize interpolation
    prep = linear_interp.interp_1d_prep(idx.size)
    c_interp = np.zeros((idx.size,D.size)) # note c_interp and v_interp are transposed of each other
    v_interp = np.zeros((D.size,idx.size))

    # b. sort m (so interp is faster)
    idx_unsort = np.argsort(np.argsort(m[idx,t])) # index to unsort 
    m_sort = np.sort(m[idx,t])

    # c. interpolate and sort back
    if ds == 1:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[:],c_sol[d],m_sort,c_interp[:,d])        
            linear_interp.interp_1d_vec_mon(prep,m_sol[:],v_sol[d],m_sort,v_interp[d])         

        # sort back
        c_interp = c_interp[idx_unsort]
        v_interp = v_interp[:,idx_unsort]

    elif ds == 0:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[:],c_sol[d],m_sort,c_interp[:,d])            

        # sort back
        c_interp = c_interp[idx_unsort]

    # d. return
    return c_interp,v_interp

@njit(parallel=True)
def optimal_choices(t,ds,c,d,probs,c_interp,v_interp,sim,par,idx,ad=0,ad_min=0):
    """ find optimal consumption and retirement choice (ad and ad_min are only so the couple model can look up in this function)"""

    # unpack
    t_idx = t + 1 + ad + ad_min
    choiceP = sim.choiceP[idx,t_idx-1,0]

    # working
    if ds == 1:
        
        # a. retirement prob and optimal labor choice
        prob = funs.logsum2(v_interp,par)[1][0] # probs are in 1 and retirement probs are in 0 
        work_choice = (prob <= choiceP[:])
        work_idx = idx[work_choice]
        ret_idx = idx[~work_choice]

        # b. optimal choices
        if t+1 < par.Tr-1:
            fill_arr(c[:,t],work_idx,c_interp[work_choice,1])
            fill_arr(c[:,t],ret_idx,c_interp[~work_choice,0])            
        else:
            fill_arr(c[:,t],idx,c_interp[:,0])

        if t+1 < par.simT:
            if t+1 < par.Tr-1:
                fill_number(d[:,t_idx],work_idx,1)
                fill_number(d[:,t_idx],ret_idx,0)  
                fill_arr(probs[:,t_idx],idx,prob)
            else:
                fill_number(d[:,t_idx],idx,0)
                fill_number(probs[:,t_idx],idx,1)
    
    # retired
    elif ds == 0:

        # a. optimal choices
        fill_arr(c[:,t],idx,c_interp[:,0])
        if t+1 < par.simT:
            fill_number(d[:,t_idx],idx,0)
            fill_number(probs[:,t_idx],idx,0)   

@njit(parallel=True)
def euler_error(t,ma,st,ra,euler,sol,par,sim,idx,ds):
    """ compute euler errors for single model"""  

    if idx.size > 0:

        # unpack
        sol_m = sol.m[:]
        sol_c = sol.c[:,ma,st]
        sol_v = sol.v[:,ma,st]
        c = sim.c[:,t]
        m = sim.m[:,t]
        a = sim.a[:,t]
        tol = par.tol
        pi_plus = transitions.survival_lookup_single(t+1,ma,st,par)    

        # 1. indices
        idx_in = idx[(tol < c[idx]) & (c[idx] < m[idx] - tol)]  # inner solution

        # 2. prep
        if ds == 1:
            D = np.array([0,1])
        elif ds == 0:
            D = np.array([0])
        idx_unsort = np.argsort(np.argsort(a[idx_in]))  # indices to unsort a
        a_sort = np.sort(a[idx_in])                     # sort a since post_decision.compute assumes a is monotone

        # 3. lhs and rhs
        avg_marg_u_plus = post_decision.compute(t,ma,st,ra,D,sol_c,sol_m,sol_v,a_sort,par)[1]
        lhs = utility.marg_func(c[idx_in],par)
        rhs = par.beta*(par.R*pi_plus*np.take(avg_marg_u_plus[ds],idx_unsort) + (1-pi_plus)*par.gamma)
        fill_arr(euler[:,t], idx_in, lhs-rhs)                 

@njit(parallel=True)
def lifecycle_c(sim,sol,single_sol,par,single_par):
    """ simulate couple model """

    # unpack
    d_w = sim.d[:,:,0]
    d_h = sim.d[:,:,1]
    RA_w = sim.RA[:,0]
    RA_h = sim.RA[:,1]
    alive_w = sim.alive[:,:,0]
    alive_h = sim.alive[:,:,1]    
    probs_w = sim.probs[:,:,0]
    probs_h = sim.probs[:,:,1]

    # states which are fixed (parallel outside loop)
    AD = sim.states[:,0]
    ST_h = sim.states[:,1]  # this has to follow the solution.solve_c (a bit backwards here compared to the rest in simulate)
    ST_w = sim.states[:,2]  # and also has to follow the xlsx (couple_formue)
    it = par.iterator
    for i in prange(len(it)):
        ad = it[i,0]
        st_w = it[i,1]
        st_h = it[i,2]
        idx_st = np.nonzero((AD==ad) & (ST_h==st_h) & (ST_w==st_w))[0]
        elig_h = transitions.state_translate(st_h,'elig',par)
        elig_w = transitions.state_translate(st_w,'elig',par)        

        for t in range(par.simT):
            tw_idx = t+ad+par.ad_min
            th_idx = t+par.ad_min

            # initialize d
            if t == 0:
                d_w[idx_st,tw_idx] = 1
                d_h[idx_st,th_idx] = 1

            # loop over retirement status
            for ra_h in np.array([0,1,2]):
                for ra_w in np.array([0,1,2]):
                    idx_ra = idx_st[(RA_h[idx_st]==ra_h) & (RA_w[idx_st]==ra_w)]
                        
                    # loop over alive status
                    for al_h in [0,1]:
                        for al_w in [0,1]:

                            # only simulate if at least one in the household is alive
                            if al_h == 1 or al_w == 1:
                                idx_alive = idx_ra[(alive_h[idx_ra,th_idx]==al_h) & (alive_w[idx_ra,tw_idx]==al_w)]

                                # loop over labor market status
                                for dh in [0,1]:
                                    for dw in [0,1]:
                                                
                                        # if both are alive we get 4 slices (4 different labor market status)
                                        if al_h == 1 and al_w == 1:
                                            idx = idx_alive[(d_h[idx_alive,th_idx]==dh) & (d_w[idx_alive,tw_idx]==dw)]
                                            simulate_couple(t,ad,st_h,st_w,elig_h,elig_w,ra_h,ra_w,sim.c,sim.m,sim.a,d_h,d_w,probs_h,probs_w,RA_h,RA_w,sim.GovS,
                                                            sol,single_sol,par,single_par,sim,
                                                            idx,dh,dw)

                                        # if only husband is alive we get 2 slices (2 different labor market status)
                                        elif al_h == 1:
                                            idx = idx_alive[(d_h[idx_alive,th_idx]==dh)]
                                            simulate_single(t,1,st_h,elig_h,ra_h,sim.c,sim.m,sim.a,d_h,probs_h,RA_h,sim.GovS,single_sol,par,sim,
                                                            idx,dh,ad=0,ad_min=par.ad_min)

                                        # if only wife is alive we get 2 slices (2 different labor market status)
                                        elif al_w == 1:
                                            idx = idx_alive[(d_w[idx_alive,tw_idx]==dw)]
                                            simulate_single(t,0,st_w,elig_w,ra_w,sim.c,sim.m,sim.a,d_w,probs_w,RA_w,sim.GovS,single_sol,par,sim,
                                                            idx,dw,ad=ad,ad_min=par.ad_min)

@njit(parallel=True)
def simulate_couple(t,ad,st_h,st_w,elig_h,elig_w,ra_h,ra_w,c,m,a,d_h,d_w,probs_h,probs_w,RA_h,RA_w,GovS,
                    sol,single_sol,par,single_par,sim,idx,dh,dw):
    """ simulate for couples """

    if idx.size > 0:
        if t > 0:
            update_m_c(t,ad,st_h,st_w,ra_h,ra_w,dh,dw,m,sim,par,idx,GovS)        
        c_interp,v_interp = ConsValue_c(t,ad,st_h,st_w,ra_h,ra_w,dh,dw,m,sol,par,idx)
        optimal_choices_c(t,ad,dh,dw,c,d_h,d_w,probs_h,probs_w,c_interp,v_interp,sim,par,idx)
        if dh == 1:
            update_ra(t,elig_h,RA_h,par,idx)
        if dw == 1:
            update_ra(t+ad,elig_w,RA_w,par,idx) 

        fill_arr(a[:,t],idx,m[idx,t]-c[idx,t])                                     

@njit(parallel=True)
def update_m_c(t,ad,st_h,st_w,ra_h,ra_w,d_h,d_w,m,sim,par,idx,GovS):
    """ update m for couples """

    # unpack
    ad_min = par.ad_min
    a = sim.a[idx,t-1]

    # both working
    if d_h == 1 and d_w == 1:
        inc = sim.labor_post_joint[idx,t]

    # husband working
    elif d_h == 1 and d_w == 0:
        pre_h = sim.labor_pre[idx,t+ad_min,1]
        pre_w = np.zeros(pre_h.shape)
        if t+ad >= par.T_oap:
            pre_w[:] = transitions.oap_pretax(t+ad,par,i=1,y=pre_w,y_spouse=pre_h)[0]
        elif t+ad >= par.T_erp:
            pre_w[:] = transitions.erp_pretax(t+ad,0,st_w,ra_w,par)[0]
        inc = (transitions.posttax(t,par,d_h,inc=pre_h,pens=np.zeros(pre_h.shape),spouse_inc=pre_w) + 
               transitions.posttax(t+ad,par,d_w,inc=np.zeros(pre_w.shape),pens=pre_w,spouse_inc=pre_h))

    # wife working
    elif d_h == 0 and d_w == 1:
        pre_w = sim.labor_pre[idx,t+ad+ad_min,0]
        pre_h = np.zeros(pre_w.shape)
        if t >= par.T_oap:
            pre_h[:] = transitions.oap_pretax(t,par,i=1,y=pre_h,y_spouse=pre_w)[0]
        elif t >= par.T_erp:
            pre_h[:] = transitions.erp_pretax(t,1,st_h,ra_h,par)[0]
        inc = (transitions.posttax(t+ad,par,d_w,inc=pre_w,pens=np.zeros(pre_w.shape),spouse_inc=pre_h) + 
               transitions.posttax(t,par,d_h,inc=np.zeros(pre_h.shape),pens=pre_h,spouse_inc=pre_w))

    # both retired
    elif d_h == 0 and d_w == 0:
        
        # husband
        pre_h = np.zeros(len(idx))
        if t >= par.T_oap:
            if t+ad < par.T_oap:
                pre_h[:] = transitions.oap_pretax(t,par,i=1)[0]
            else:
                pre_h[:] = transitions.oap_pretax(t,par,i=2)[0]
        elif par.T_erp <= t < par.T_oap:
            pre_h[:] = transitions.erp_pretax(t,1,st_h,ra_h,par)[0]

        # wife
        pre_w = np.zeros(len(idx))
        if t+ad >= par.T_oap:
            if t < par.T_oap:
                pre_w[:] = transitions.oap_pretax(t+ad,par,i=1)[0]
            else:
                pre_w[:] = transitions.oap_pretax(t+ad,par,i=2)[0]
        elif par.T_erp <= t+ad < par.T_oap:
            pre_w[:] = transitions.erp_pretax(t+ad,0,st_w,ra_w,par)[0]
        
        # post tax
        inc = (transitions.posttax(t,par,d_h,inc=np.zeros(len(idx)),pens=pre_h,spouse_inc=pre_w) + 
               transitions.posttax(t+ad,par,d_w,inc=np.zeros(len(idx)),pens=pre_w,spouse_inc=pre_h))

    # update m
    fill_arr(m[:,t],idx,par.R*a[:] + inc[:])

    # government surplus
    if sim.tax:
        if d_h == 1 and d_w == 1:
            pre = sim.labor_pre_joint[idx,t]
        else:
            pre = d_w*pre_w[:] + d_h*pre_h[:]
        fill_arr(GovS[:,t],idx, pre[:] - inc[:]) 

@njit(parallel=True)
def ConsValue_c(t,ad,st_h,st_w,ra_h,ra_w,d_h,d_w,m,sol,par,idx):
    """ interpolate consumption and value for couple model """

    # a. unpack solution
    D = transitions.d_plus_c(t-1,ad,d_h,d_w,par)    # t-1 so we get choice set today
    ra_look_h = transitions.ra_look_up(t,st_h,ra_h,d_h,par)
    ra_look_w = transitions.ra_look_up(t+ad,st_w,ra_w,d_w,par)
    ad_idx = ad+par.ad_min
    c_sol = sol.c[t,ad_idx,st_h,st_w,ra_look_h,ra_look_w]
    m_sol = sol.m[:]
    v_sol = sol.v[t,ad_idx,st_h,st_w,ra_look_h,ra_look_w] 

    # a. initialize interpolation
    prep = linear_interp.interp_1d_prep(idx.size)
    c_interp = np.zeros((idx.size,4))   # note c_interp and v_interp are transposed of each other
    v_interp = np.zeros((4,idx.size)) 

    # b. sort m (so interp is faster)
    idx_unsort = np.argsort(np.argsort(m[idx,t])) # index to unsort 
    m_sort = np.sort(m[idx,t])                

    # c. interpolate and sort back
    if d_h == 0 and d_w == 0:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[:],c_sol[d],m_sort,c_interp[:,d])        

        # sort back
        c_interp = c_interp[idx_unsort]

    else:
        for d in D:
            linear_interp.interp_1d_vec_mon(prep,m_sol[:],c_sol[d],m_sort,c_interp[:,d])        
            linear_interp.interp_1d_vec_mon(prep,m_sol[:],v_sol[d],m_sort,v_interp[d])         

        # sort back
        c_interp = c_interp[idx_unsort]
        v_interp = v_interp[:,idx_unsort]    
    
    # d. return
    return c_interp,v_interp

@njit(parallel=True)
def optimal_choices_c(t,ad,dh_t,dw_t,c,d_h,d_w,probs_h,probs_w,c_interp,v_interp,sim,par,idx):
    """ optimal choices for couples """

    # unpack
    th_idx = t + 1 + par.ad_min
    tw_idx = t + 1 + ad + par.ad_min
    choiceP_w = sim.choiceP[idx,th_idx-1,0]
    choiceP_h = sim.choiceP[idx,tw_idx-1,1]    

    # both work
    if dh_t == 1 and dw_t == 1:

        # retirement probs and choices
        prob = funs.logsum4(v_interp,par)[1]
        prob_w = prob[0] + prob[2]
        prob_h = prob[0] + prob[1]
        work_choice_w = (prob_w <= choiceP_w[:])
        work_choice_h = (prob_h <= choiceP_h[:])
        work_idx_w = idx[work_choice_w]
        ret_idx_w = idx[~work_choice_w]
        work_idx_h = idx[work_choice_h]
        ret_idx_h = idx[~work_choice_h]        

        # optimal choices
        if t+1 < par.simT:

            # husband
            if t+1 < par.Tr-1:
                fill_number(d_h[:,th_idx],work_idx_h,1)
                fill_number(d_h[:,th_idx],ret_idx_h,0)
                fill_arr(probs_h[:,th_idx],idx,prob_h)
            else:
                fill_number(d_h[:,th_idx],idx,0)
                fill_number(probs_h[:,th_idx],idx,1)

            # wife
            if t+1+ad < par.Tr-1:            
                fill_number(d_w[:,tw_idx],work_idx_w,1)
                fill_number(d_w[:,tw_idx],ret_idx_w,0)
                fill_arr(probs_w[:,tw_idx],idx,prob_w)
            else:
                fill_number(d_w[:,tw_idx],idx,0)          
                fill_number(probs_w[:,tw_idx],idx,1)         

    # husband work
    if dh_t == 1 and dw_t == 0:

        # retirement probs and choices
        prob = funs.logsum4(v_interp,par)[1]
        prob_h = prob[0] + prob[1]
        work_choice_h = ((prob_h <= choiceP_h[:]))
        work_idx_h = idx[work_choice_h]
        ret_idx_h = idx[~work_choice_h]        

        # optimal choices
        if t+1 < par.simT:
            
            # husband
            if t+1 < par.Tr-1:
                fill_number(d_h[:,th_idx],work_idx_h,1)
                fill_number(d_h[:,th_idx],ret_idx_h,0)
                fill_arr(probs_h[:,th_idx],idx,prob_h)
            else:
                fill_number(d_h[:,th_idx],idx,0)
                fill_number(probs_h[:,th_idx],idx,1)

            # wife
            fill_number(d_w[:,tw_idx],idx,0)
            fill_number(probs_w[:,tw_idx],idx,0)

    # wife work
    if dh_t == 0 and dw_t == 1:

        # retirement probs and choices
        prob = funs.logsum4(v_interp,par)[1]
        prob_w = prob[0] + prob[2]
        work_choice_w = ((prob_w <= choiceP_w[:]))
        work_idx_w = idx[work_choice_w]
        ret_idx_w = idx[~work_choice_w]

        # optimal choices
        if t+1 < par.simT:
            
            # wife
            if t+1+ad < par.Tr-1:            
                fill_number(d_w[:,tw_idx],work_idx_w,1)
                fill_number(d_w[:,tw_idx],ret_idx_w,0)         
                fill_arr(probs_w[:,tw_idx],idx,prob_w)
            else:
                fill_number(d_w[:,tw_idx],idx,0) 
                fill_number(probs_w[:,tw_idx],idx,1)           

            # wife
            fill_number(d_h[:,th_idx],idx,0)
            fill_number(probs_h[:,th_idx],idx,0)

    # both retired
    if dh_t == 0 and dw_t == 0:
        if t+1 < par.simT:
            
            # husband
            fill_number(d_h[:,th_idx],idx,0)
            fill_number(probs_h[:,th_idx],idx,0)

            # wife
            fill_number(d_w[:,tw_idx],idx,0)            
            fill_number(probs_w[:,tw_idx],idx,0)

    # optimal consumption
    if t+1 < par.simT:
        dw = d_w[idx,tw_idx]
        dh = d_h[idx,th_idx]
        d0 = ((dh == 0) & (dw == 0))
        d1 = ((dh == 0) & (dw == 1))
        d2 = ((dh == 1) & (dw == 0))
        d3 = ((dh == 1) & (dw == 1))
        fill_arr(c[:,t],idx[d0],c_interp[:,0])
        fill_arr(c[:,t],idx[d1],c_interp[:,1])
        fill_arr(c[:,t],idx[d2],c_interp[:,2])
        fill_arr(c[:,t],idx[d3],c_interp[:,3])

# @njit(parallel=True)
# def euler_error(t,ma,st,ra,euler,sol,par,sim,idx,ds):
#     """ compute euler errors for single model"""  

#     if idx.size > 0:

#         # unpack
#         sol_m = sol.m[:]
#         sol_c = sol.c[:,ma,st]
#         sol_v = sol.v[:,ma,st]
#         c = sim.c[:,t]
#         m = sim.m[:,t]
#         a = sim.a[:,t]
#         tol = par.tol
#         pi_plus = transitions.survival_lookup_single(t+1,ma,st,par)    

#         # 1. indices
#         idx_in = idx[(tol < c[idx]) & (c[idx] < m[idx] - tol)]  # inner solution

#         # 2. prep
#         if ds == 1:
#             D = np.array([0,1])
#         elif ds == 0:
#             D = np.array([0])
#         idx_unsort = np.argsort(np.argsort(a[idx_in]))  # indices to unsort a
#         a_sort = np.sort(a[idx_in])                     # sort a since post_decision.compute assumes a is monotone

#         # 3. lhs and rhs
#         avg_marg_u_plus = post_decision.compute(t,ma,st,ra,D,sol_c,sol_m,sol_v,a_sort,par)[1]
#         lhs = utility.marg_func(c[idx_in],par)
#         rhs = par.beta*(par.R*pi_plus*np.take(avg_marg_u_plus[ds],idx_unsort) + (1-pi_plus)*par.gamma)
#         fill_arr(euler[:,t], idx_in, lhs-rhs)                 


# @njit(parallel=True)
# def euler_error_c(t,ad,st_h,st_w,ra_h,ra_w,euler,
#                   sim,sol,par,single_sol,single_par,
#                   work,H_work,W_work,ret,
#                   H_alive_work,H_alive_ret,W_alive_work,W_alive_ret):
#     """ compute euler errors for couple model"""  

#     # unpack
#     single_sol_ch = single_sol.c[:,0,1,st_h]
#     single_sol_mh = single_sol.m[:,0,1,st_h]
#     single_sol_vh = single_sol.v[:,0,1,st_h]
#     single_sol_cw = single_sol.c[:,0,0,st_w]
#     single_sol_mw = single_sol.m[:,0,0,st_w]
#     single_sol_vw = single_sol.v[:,0,0,st_w]    
#     c = sim.c[:,t]
#     m = sim.m[:,t]
#     a = sim.a[:,t]
#     tol = par.tol
#     ad_min = par.ad_min
#     pi_plus_h,pi_plus_w = transitions.survival_look_up_c(t+1,ad,par)    

#     # 1. indices
#     work_in = work[(tol < c[work]) & (c[work] < m[work] - tol)]
#     H_work_in = work[(tol < c[H_work]) & (c[H_work] < m[H_work] - tol)]
#     W_work_in = work[(tol < c[W_work]) & (c[W_work] < m[W_work] - tol)]
#     ret_in = work[(tol < c[ret]) & (c[ret] < m[ret] - tol)]
#     Hal_work_in = work[(tol < c[H_alive_work]) & (c[H_alive_work] < m[H_alive_work] - tol)]
#     Hal_ret_in = work[(tol < c[H_alive_ret]) & (c[H_alive_ret] < m[H_alive_ret] - tol)]
#     Wal_work_in = work[(tol < c[W_alive_work]) & (c[W_alive_work] < m[W_alive_work] - tol)]
#     Wal_ret_in = work[(tol < c[W_alive_ret]) & (c[W_alive_ret] < m[W_alive_ret] - tol)]                

#     # 2. both work
#     if work_in.size > 0:

#         # choice set and sort a (post_decision.compute_c assumes a is monotone)
#         D_h = np.array([0,1])
#         D_w = np.array([0,1])
#         d = transitions.d_c(1,1)        
#         idx_unsort = np.argsort(np.argsort(a[work_in]))
#         a_sort = np.sort(a[work_in])

#         # lhs and rhs      
#         q = post_decision.compute_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a_sort,
#                                     sol.c,sol.m,sol.v,
#                                     single_sol.v_plus_raw,single_sol.avg_marg_u_plus,look_up=False)[1]
#         rhs = np.take(q[d],idx_unsort)
#         lhs = utility.marg_func(c[work_in],par)
#         fill_arr(euler[:,t], work_in, lhs-rhs)

#     # 3. husband work
#     if H_work_in.size > 0:

#         # choice set and sort a (post_decision.compute_c assumes a is monotone)
#         D_h = np.array([0,1])
#         D_w = np.array([0])
#         d = transitions.d_c(1,0)        
#         idx_unsort = np.argsort(np.argsort(a[H_work_in]))
#         a_sort = np.sort(a[H_work_in])

#         # lhs and rhs      
#         q = post_decision.compute_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a_sort,
#                                     sol.c,sol.m,sol.v,
#                                     single_sol.v_plus_raw,single_sol.avg_marg_u_plus,look_up=False)[1]
#         rhs = np.take(q[d],idx_unsort)
#         lhs = utility.marg_func(c[H_work_in],par)
#         fill_arr(euler[:,t], H_work_in, lhs-rhs)        

#     # 4. wife work
#     if W_work_in.size > 0:

#         # choice set and sort a (post_decision.compute_c assumes a is monotone)
#         D_h = np.array([0])
#         D_w = np.array([0,1])
#         d = transitions.d_c(0,1)
#         idx_unsort = np.argsort(np.argsort(a[W_work_in]))
#         a_sort = np.sort(a[W_work_in])

#         # lhs and rhs      
#         q = post_decision.compute_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a_sort,
#                                     sol.c,sol.m,sol.v,
#                                     single_sol.v_plus_raw,single_sol.avg_marg_u_plus,look_up=False)[1]
#         rhs = np.take(q[d],idx_unsort)
#         lhs = utility.marg_func(c[W_work_in],par)
#         fill_arr(euler[:,t], W_work_in, lhs-rhs)        

#     # 5. both retired
#     if ret_in.size > 0:

#         # choice set and sort a (post_decision.compute_c assumes a is monotone)
#         D_h = np.array([0])
#         D_w = np.array([0])
#         d = transitions.d_c(0,0)        
#         idx_unsort = np.argsort(np.argsort(a[ret_in]))
#         a_sort = np.sort(a[ret_in])

#         # lhs and rhs      
#         q = post_decision.compute_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a_sort,
#                                     sol.c,sol.m,sol.v,
#                                     single_sol.v_plus_raw,single_sol.avg_marg_u_plus,look_up=False)[1]
#         rhs = np.take(q[d],idx_unsort)
#         lhs = utility.marg_func(c[ret_in],par)
#         fill_arr(euler[:,t], ret_in, lhs-rhs)      

#     # 6. husband alive and work
#     if Hal_work_in.size > 0:

#         # choice set and sort a (post_decision.compute assumes a is monotone)
#         D = np.array([0,1])
#         idx_unsort = np.argsort(np.argsort(a[Hal_work_in]))
#         a_sort = np.sort(a[Hal_work_in])        

#         # lhs and rhs
#         avg_marg_u_plus = post_decision.compute(t+ad_min,0,1,st_h,ra_h,D,single_sol_ch,single_sol_mh,single_sol_vh,a_sort,single_par,look_up=False)[1]
#         lhs = utility.marg_func(c[Hal_work_in],par)
#         rhs = par.beta*(par.R*pi_plus_h*np.take(avg_marg_u_plus[1],idx_unsort) + (1-pi_plus_h)*par.gamma)
#         fill_arr(euler[:,t], Hal_work_in, lhs-rhs)

#     # 7. husband alive and retired
#     if Hal_ret_in.size > 0:

#         # choice set and sort a (post_decision.compute assumes a is monotone)
#         D = np.array([0])
#         idx_unsort = np.argsort(np.argsort(a[Hal_ret_in]))
#         a_sort = np.sort(a[Hal_ret_in])

#         # lhs and rhs
#         avg_marg_u_plus = post_decision.compute(t+ad_min,0,1,st_h,ra_h,D,single_sol_ch,single_sol_mh,single_sol_vh,a_sort,single_par,look_up=False)[1]
#         lhs = utility.marg_func(c[Hal_ret_in],par)
#         rhs = par.beta*(par.R*pi_plus_h*np.take(avg_marg_u_plus[0],idx_unsort) + (1-pi_plus_h)*par.gamma)
#         fill_arr(euler[:,t], Hal_ret_in, lhs-rhs)

#     # 8. wife alive and work
#     if Wal_work_in.size > 0:

#         # choice set and sort a (post_decision.compute assumes a is monotone)
#         D = np.array([0,1])
#         idx_unsort = np.argsort(np.argsort(a[Wal_work_in]))
#         a_sort = np.sort(a[Wal_work_in])        

#         # lhs and rhs
#         avg_marg_u_plus = post_decision.compute(t+ad+ad_min,0,0,st_w,ra_w,D,single_sol_cw,single_sol_mw,single_sol_vw,a_sort,single_par,look_up=False)[1]
#         lhs = utility.marg_func(c[Wal_work_in],par)
#         rhs = par.beta*(par.R*pi_plus_w*np.take(avg_marg_u_plus[1],idx_unsort) + (1-pi_plus_w)*par.gamma)
#         fill_arr(euler[:,t], Wal_work_in, lhs-rhs)

#     # 9. wife alive and retired
#     if Wal_ret_in.size > 0:

#         # choice set and sort a (post_decision.compute assumes a is monotone)
#         D = np.array([0])
#         idx_unsort = np.argsort(np.argsort(a[Wal_ret_in]))
#         a_sort = np.sort(a[Hal_ret_in])

#         # lhs and rhs
#         avg_marg_u_plus = post_decision.compute(t+ad+ad_min,0,0,st_w,ra_w,D,single_sol_cw,single_sol_mw,single_sol_vw,a_sort,single_par,look_up=False)[1]
#         lhs = utility.marg_func(c[Wal_ret_in],par)
#         rhs = par.beta*(par.R*pi_plus_w*np.take(avg_marg_u_plus[1],idx_unsort) + (1-pi_plus_w)*par.gamma)
#         fill_arr(euler[:,t], Wal_ret_in, lhs-rhs)