# global modules
from numba import njit, prange
import numpy as np

# consav
from consav import linear_interp

# local modules 
import post_decision
import egm
import transitions
import last_period
import funs

@njit(parallel=True)
def ret_dict(t,par):
    """ function which mimics a dict, returns information on how to resolve the model for retirement status"""

    if t == par.T_oap-1:
        ra = np.array([0,1,2])
        d = np.array([1,0,0])
    elif t == par.T_two_year+1:
        ra = np.array([0,1,2])
        d = np.array([1,0,0])
    elif t == par.T_two_year:
        ra = np.array([0,1,2])
        d = np.array([1,0,0])        
    elif t == par.T_two_year-1:
        ra = np.array([1,2])
        d = np.array([1,0])        
    elif t == par.T_erp:
        ra = np.array([1,2])  
        d = np.array([1,0])      
    elif t == par.T_erp-1:
        ra = np.array([2]) 
        d = np.array([1])

    return ra.size,ra,d  

@njit(parallel=True)
def solve(sol,par):
    """ wrapper for solving the single model"""
    it = par.iterator
    for j in prange(len(it)):
        ma = it[j,0]
        st = it[j,1]

        # unpack solution
        sol_c = sol.c[:,ma,st]
        sol_m = sol.m
        sol_v = sol.v[:,ma,st]
        sol_v_plus_raw = sol.v_plus_raw[:,ma,st]
        sol_avg_marg_u_plus = sol.avg_marg_u_plus[:,ma,st]
        a = par.grid_a
                
        # solve
        solve_single_model(ma,st,sol_c,sol_m,sol_v,sol_v_plus_raw,sol_avg_marg_u_plus,a,par)                

@njit(parallel=True)
def solve_single_model(ma,st,sol_c,sol_m,sol_v,sol_v_plus_raw,sol_avg_marg_u_plus,a,par):

    # prep
    elig = transitions.state_translate(st,'elig',par)

    # backwards induction
    for t in range(par.T-1,-1,-1):  # same as reversed(range(par.T))

        # 1. Oap age: Solution is independent of retirement age
        if t+1 >= par.T_oap:
            if elig == 1:
                ra = 0
            else:
                ra = 2

            if t == par.T-1:    # last period
                last_period.solve(t,ma,st,ra,0,sol_c,sol_m,sol_v,par)
                        
            elif t+1 >= par.Tr: # forced to retire
                D = np.array([0])
                egm.solve_bellman(t,ma,st,ra,D,sol_c,sol_m,sol_v,sol_v_plus_raw,sol_avg_marg_u_plus,a,par)

            else:               # not forced to retire
                D = np.array([0,1])
                egm.solve_bellman(t,ma,st,ra,D,sol_c,sol_m,sol_v,sol_v_plus_raw,sol_avg_marg_u_plus,a,par)

        # 2. Erp age and eligible: Solution depends of retirement age
        elif (t+1 >= par.T_erp-1 and elig == 1):
            ret = ret_dict(t+1,par)
            for rs in range(ret[0]):
                ra = ret[1][rs]         # retirement age
                if ret[2][rs] == 1:     # choice set
                    D = np.array([0,1])
                elif ret[2][rs] == 0:
                    D = np.array([0]) 

                egm.solve_bellman(t,ma,st,ra,D,sol_c,sol_m,sol_v,sol_v_plus_raw,sol_avg_marg_u_plus,a,par)

        # 3. Pre Erp age or not eligible: Solution is independent of retirement age
        else:
            ra = 2
            D = np.array([0,1])
            egm.solve_bellman(t,ma,st,ra,D,sol_c,sol_m,sol_v,sol_v_plus_raw,sol_avg_marg_u_plus,a,par)

@njit(parallel=True)
def solve_c(sol,single_sol,par):
    """ wrapper for solving the couple model"""

    it = par.iterator
    for j in prange(len(it)):
        ad = it[j,0]
        st_h = it[j,1]
        st_w = it[j,2]

        # solve
        solve_couple_model(ad,st_h,st_w,par,par.grid_a,
                           sol.c,sol.m,sol.v,
                           single_sol.v_plus_raw,single_sol.avg_marg_u_plus)                               

@njit(parallel=True)
def solve_couple_model(ad,st_h,st_w,par,a,
                       sol_c,sol_m,sol_v,
                       single_sol_v_plus_raw,single_sol_avg_marg_u_plus):

    # eligibility to erp
    elig_h = transitions.state_translate(st_h,'elig',par)
    elig_w = transitions.state_translate(st_w,'elig',par)  

    # backwards induction
    for t in range(par.T-1,-1,-1):  # same as reversed(range(par.T))        
                                 
        # 1. Oap age: Solution is independent of retirement age
        if min(t+1,t+1+ad) >= par.T_oap:
            ra_h = transitions.ra_look_up(t,st_h,0,0,par)   # ra=0 and d=0, doesn't matter here, but have to fill them out
            ra_w = transitions.ra_look_up(t+ad,st_w,0,0,par)   # ra=0 and d=0                        

            if t == par.T-1:    # last period
                last_period.solve_c(t,ad,st_h,st_w,ra_h,ra_w,0,0,sol_c,sol_m,sol_v,par)   # d_h=0, d_w=0

            else:               # not last period
                # choice set husband
                if t+1 >= par.Tr:
                    D_h = np.array([0])
                else:
                    D_h = np.array([0,1])

                # choice set wife
                if t+1+ad >= par.Tr:
                    D_w = np.array([0])
                else:
                    D_w = np.array([0,1])

                # solve
                egm.solve_bellman_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
                                    sol_c,sol_m,sol_v,
                                    single_sol_v_plus_raw,single_sol_avg_marg_u_plus)

        # 2. Pre Oap age: Solution potentially depends on retirement age
        else:

            # if we need to recalculate solution, husband
            if par.T_erp-1 <= t+1 <= par.T_oap-1 and elig_h == 1:
                ret_h = ret_dict(t+1,par)
                for rh in range(ret_h[0]):
                    ra_h = ret_h[1][rh]
                    if ret_h[2][rh] == 1:
                        D_h = np.array([0,1])
                    else:
                        D_h = np.array([0])
                
                    # if we need to recalculate solution, wife
                    if par.T_erp-1 <= t+1+ad <= par.T_oap-1 and elig_w == 1:
                        ret_w = ret_dict(t+1+ad,par)
                        for rw in range(ret_w[0]):
                            ra_w = ret_w[1][rw]
                            if ret_w[2][rw] == 1:
                                D_w = np.array([0,1])
                            else:
                                D_w = np.array([0])

                            egm.solve_bellman_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
                                                sol_c,sol_m,sol_v,
                                                single_sol_v_plus_raw,single_sol_avg_marg_u_plus)
                                
                    # don't need to recalculate for wife (but still do for husband)
                    else:
                        ra_w = transitions.ra_look_up(t+ad,st_w,0,1,par)
                        D_w = np.array([0,1])
                        egm.solve_bellman_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
                                            sol_c,sol_m,sol_v,
                                            single_sol_v_plus_raw,single_sol_avg_marg_u_plus)
                        
            # if we don't need to recalculate solution, husband
            else:
                ra_h = transitions.ra_look_up(t,st_h,0,1,par)
                D_h = np.array([0,1])

                # if we need to recalculate solution, wife
                if par.T_erp-1 <= t+1+ad <= par.T_oap-1 and elig_w == 1:
                        ret_w = ret_dict(t+1+ad,par)
                        for rw in range(ret_w[0]):
                            ra_w = ret_w[1][rw]
                            if ret_w[2][rw] == 1:
                                D_w = np.array([0,1])
                            else:
                                D_w = np.array([0])
                                
                            egm.solve_bellman_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
                                                sol_c,sol_m,sol_v,
                                                single_sol_v_plus_raw,single_sol_avg_marg_u_plus) 
                            
                # don't need to recalculate solution for any of them
                else:
                    ra_w = transitions.ra_look_up(t+ad,st_w,0,1,par)
                    D_w = np.array([0,1])
                    egm.solve_bellman_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,par,a,
                                        sol_c,sol_m,sol_v,
                                        single_sol_v_plus_raw,single_sol_avg_marg_u_plus) 