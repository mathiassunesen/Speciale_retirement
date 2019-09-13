# global modules
from numba import njit, prange
import numpy as np

# local modules 
import egm
import transitions
import last_period


@njit(parallel=True)
def solve(par,sol):
    """ solve the model after the solution have been allocated 
    
        Args:
        
            par=parameters, sol=solution"""

    for ad in range(len(par.age_dif)):     # loop over age differences for couples
        for st in range(len(par.states)):  # loop over states (solve model many times)
            for t in range(par.T-1,-1,-1): # same as reversed(range(par.T))

                # i. last period
                if t == par.T-1:
                    last_period.solve(t,st,ad,sol,par)

                # ii. if forced to retire (retirement decision is made one period ahead)
                elif t+1 >= par.Tr-1:
                    d = np.arange(1)    # same as [0]
                    egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,0,0])

                # iii. if not forced to retire
                else:
                    d = np.arange(2)    # same as [0,1]

                    # a. oap age
                    if transitions.age(t+1,par) >= 65:
                        egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,0,0])

                    # b. if eligible to erp we have to recaulculate solution
                    elif (59 <= transitions.age(t+1,par) <= 64 and 
                          transitions.state_translate(st,'elig',par) == 1):
                        
                        age_next = transitions.age(t+1,par)
                        if age_next == 64:
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,0,0])
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,1,1])
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,2,2])   
                        elif 62 <= age_next <= 63:
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,0,0])
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[1,1,1])
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[2,2,2]) 
                        elif age_next == 61:
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[1,1,0])
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[2,2,2])
                        elif age_next == 60:
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,1,0])
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[2,2,2])
                        elif age_next == 59:
                            egm.solve_singles(t,st,ad,d,sol,par,retirement=[2,2,0])                                                                                                                                             

                    # c. pre erp age
                    else:
                        egm.solve_singles(t,st,ad,d,sol,par,retirement=[0,2,0])                        



# @njit(parallel=True)
# def solve_couples(self,single_sol):
#         """ solve the model """

#     # prep
#     par = self.par
#     sol = self.sol

#     # a. allocate solution
#     self._solve_prep(single_sol)
        
#     # b. backwards induction
#     for ad in range(len(par.age_dif)):      # loop over age differences for couples
#         for st in range(len(par.states)):   # loop over states (solve model many times)
#             for t in range(par.T-1,-1,-1):  # same as reversed(range(par.T))       

#                 # i. last period
#                 if t == par.T-1:
#                     last_period.solve(t,st,ad,sol,par)

#                 # ii. if both are forced to retire (retirement decision is made one period ahead)
#                 elif max(t+1,t+1+ad) >= par.Tr-1:   # ad can be both negative and positive
#                     d = [0] 
#                     egm.solve_couples(t,st,ad,d,sol,par,retirement=[0,0,0])                    

#                 # iii. if husband is forced to retire, but not wife
#                 elif t+1 >= par.Tr-1 and t+1+ad < par.Tr-1:
#                     d = [0,1]
#                     egm.solve_couples(t,st,ad,d,sol,par,retirement=[0,0,0])
                    
#                 # iv. if wife is forced to retire but not husband
#                 elif t+1 < par.Tr-1 and t+1+ad >= par.Tr-1:
#                     d = [0,2]
#                     egm.solve_couples(t,st,ad,d,sol,par,retirement=[0,0,0])   

#                 # v. if none are forced to retire
#                 else:
#                     d = [0,1,2,3]

#                     # now we check if we have to recalculate solution
#                     h_age_next = transitions.age(t+1,par)    # husband age in next period
#                     w_age_next = h_age_next + ad                  # wife age in next period

#                     # a. both in oap age
#                     if max(h_age_next,w_age_next) >= oap_age:
#                         retirement = [[0,0,0], [0,0,0]]  # no need to recalculate solution

#                         # b. both in erp age
#                     elif erp_age <= h_age_next < oap_age and erp_age <= w_age_next < oap_age:
#                         retirement = [par.ret_system[h_age_next], par.ret_system[w_age_next]]

#                     # b. husband in oap age, but wife in erp age
#                     elif (h_age_next >= oap_age) and (erp_age <= w_age_next < oap_age):
#                         retirement = [[0,0,0], par.ret_system[w_age_next]]

#                     # c. husband in erp age, but wife in oap age
#                     elif erp_age <= h_age_next < oap_age and w_age_next >= oap_age:
#                         retirement = [par.ret_system[h_age_next], [0,0,0]]

#                     # both pre erp age
#                     else:
#                         retirement = [[0,2,0], [0,2,0]]

#                     # vi. solve the model
#                     for ir in retirement:
#                         egm.solve_couples(t,st,ad,d,sol,par,ir)                                           