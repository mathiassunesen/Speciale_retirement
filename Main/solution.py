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
    for ad in par.age_dif:                      # loop over age differences (just zero here)
        for ma in range(len(par.male)):         # loop over gender
            for st in range(len(par.states)):   # loop over states
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
                            d_lst = np.array([0])
                            egm.solve(t,ad,ma,st,ra,d_lst,sol,par)

                        else:               # not forced to retire
                            d_lst = np.array([0,1])
                            egm.solve(t,ad,ma,st,ra,d_lst,sol,par)

                    # 2. Erp age and eligible: Solution depends of retirement age
                    elif (t+1 >= erp_age-1 and 
                          transitions.state_translate(st,'elig',par) == 1):
                        for ir in ret_system[t+1]:  # ret_system contains information on how to recalculate solution
                            ra = ir[0]
                            d_lst = ir[1]
                            egm.solve(t,ad,ma,st,ra,d_lst,sol,par)

                    # 3. Pre Erp age or not eligible: Solution is independent of retirement age
                    else:
                        ra = 2
                        d_lst = np.array([0,1])
                        egm.solve(t,ad,ma,st,ra,d_lst,sol,par)




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
    for ad in par.age_dif:                          # loop over age differences for couples
        for st_h in range(len(par.states)):         # loop over states for husband
            for st_w in range(len(par.states)):     # loop over states for wife
                for t in range(par.T-1,-1,-1):      # same as reversed(range(par.T))        

                    # i. last period
                    if t == par.T-1:
                        last_period.solve_c(t,ad,st_h,st_w,0,0,sol,par)   # erp=0 and d=0

                    # ii. if both are forced to retire (retirement decision is made one period ahead)
                    elif min(t+1,t+1+ad) >= par.Tr:   # ad can be both negative and positive
                        d = np.array([0])     
                        egm.solve_c(t,ad,st_h,st_w,d,sol,par,single_sol,retirement=[0,0,0])                    

                    # iii. if husband is forced to retire, but not wife
                    elif t+1 >= par.Tr and t+1+ad < par.Tr:
                        d = np.array([0,1])
                        egm.solve_c(t,ad,st_h,st_w,d,sol,par,single_sol,retirement=[0,0,0])

                    # iii. if wife is forced to retire, but not husband
                    elif t+1 < par.Tr and t+1+ad >= par.Tr:
                        d = np.array([0,2])
                        egm.solve_c(t,ad,st_h,st_w,d,sol,par,single_sol,retirement=[0,0,0]) 

                    # iv. if none are forced to retire
                    else:
                        d = np.array([0,1,2,3])
                        egm.solve_c(t,ad,st_h,st_w,d,sol,par,single_sol,retirement=[0,0,0])                                          
