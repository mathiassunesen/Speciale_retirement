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


def solve(model):
    """ solve the model for singles after the solution have been allocated
    
    Args:
        model (class): contains par (paramenters) and sol (solution)

    Returns:
        stores solution in model.sol
    """
    # unpack 
    sol = model.sol
    par = model.par
    ret_system = model.ret_system   
    oap_age = max(ret_system)+1
    erp_age = min(ret_system)+1   

    # backwards induction
    for ad in par.AD:                           # loop over age differences (just zero here)
        for ma in range(len(par.MA)):           # loop over gender
            for st in range(len(par.ST)):       # loop over states
                for t in range(par.T-1,-1,-1):  # same as reversed(range(par.T))

                    # eligibility to erp
                    elig = transitions.state_translate(st,'elig',par)

                    # 1. Oap age: Solution is independent of retirement age
                    if t+1 >= oap_age:
                        if elig == 1:
                            ra = 0
                        else:
                            ra = 2

                        if t == par.T-1:    # last period
                            last_period.solve(t,ad,ma,st,ra,0,sol,par)
                        
                        elif t+1 >= par.Tr: # forced to retire
                            D = np.array([0])
                            egm.solve(t,ad,ma,st,ra,D,sol,par)

                        else:               # not forced to retire
                            D = np.array([0,1])
                            egm.solve(t,ad,ma,st,ra,D,sol,par)

                    # 2. Erp age and eligible: Solution depends of retirement age
                    elif (t+1 >= erp_age-1 and elig == 1):
                        for rs in ret_system[t+1]:  # ret_system contains information on how to recalculate solution
                            ra = rs[0]
                            D = rs[1]
                            egm.solve(t,ad,ma,st,ra,D,sol,par)

                    # 3. Pre Erp age or not eligible: Solution is independent of retirement age
                    else:
                        ra = 2
                        D = np.array([0,1])
                        egm.solve(t,ad,ma,st,ra,D,sol,par)




def solve_c(model):
    """ solve the model for couples after the solution have been allocated and the model for singles have been solved
    
    Args:
        model (class): contains par (parameters), sol (solution) and single_sol (single solution)

    Returns:
        stores solution in model.sol
    """
    # unpack 
    sol = model.sol
    par = model.par
    single_sol = model.Single.sol
    ret_system = model.ret_system   
    oap_age = max(ret_system)+1
    erp_age = min(ret_system)+1    

    # backwards induction
    for ad in par.AD:                           # loop over age differences for couples
        for st_h in range(len(par.ST)):         # loop over states for husband
            for st_w in range(len(par.ST)):     # loop over states for wife
                for t in range(par.T-1,-1,-1):  # same as reversed(range(par.T))        
                                 
                    # 1. Oap age: Solution is independent of retirement age
                    if min(t+1,t+1+ad) >= oap_age:
                        ra_h = transitions.ra_look_up(t,st_h,0,0,par)   # ra=0 and d=0, doesn't matter here, but have to fill them out
                        ra_w = transitions.ra_look_up(t,st_w,0,0,par)   # ra=0 and d=0                        

                        if t == par.T-1:    # last period
                            last_period.solve_c(t,ad,st_h,st_w,ra_h,ra_w,0,0,sol,par)   # d_h=0, d_w=0

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
                            egm.solve_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,sol,par,single_sol)

                    # 2. Pre Oap age: Solution potentially depends on retirement age
                    else:

                        # eligibility to erp
                        elig_h = transitions.state_translate(st_h,'elig',par)
                        elig_w = transitions.state_translate(st_w,'elig',par)   

                        # if we need to recalculate solution, husband
                        if erp_age-1 <= t+1 <= oap_age-1 and elig_h == 1:
                            for rh in ret_system[t+1]:
                                ra_h = rh[0]
                                D_h = rh[1]

                                # if we need to recalculate solution, wife
                                if erp_age-1 <= t+1+ad <= oap_age-1 and elig_w == 1:
                                    for rw in ret_system[t+1+ad]:
                                        ra_w = rw[0]
                                        D_w = rw[1]
                                        egm.solve_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,sol,par,single_sol)
                                
                                # don't need to recalculate for wife (but still do for husband)
                                else:
                                    ra_w = transitions.ra_look_up(t,st_w,0,1,par)
                                    D_w = np.array([0,1])
                                    egm.solve_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,sol,par,single_sol)   
                        
                        # if we don't need to recalculate solution, husband
                        else:
                            ra_h = transitions.ra_look_up(t,st_h,0,1,par)
                            D_h = np.array([0,1])

                            # if we need to recalculate solution, wife
                            if erp_age-1 <= t+1+ad <= oap_age-1 and elig_w == 1:
                                for rw in ret_system[t+1+ad]:
                                    ra_w = rw[0]
                                    D_w = rw[1]
                                    egm.solve_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,sol,par,single_sol)  
                            
                            # don't need to recalculate solution for any of them
                            else:
                                ra_w = transitions.ra_look_up(t,st_w,0,1,par)
                                D_w = np.array([0,1])
                                egm.solve_c(t,ad,st_h,st_w,ra_h,ra_w,D_h,D_w,sol,par,single_sol)                                                                  