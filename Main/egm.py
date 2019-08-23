from numba import njit, prange
import numpy as np

# consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope 

# local modules
import utility
import post_decision
import transitions

@njit(parallel=True)
def is_sorted(a):
    for i in range(a.size-1):
         if a[i+1] < a[i]:
               return False
    return True

@njit(parallel=True)
def solve_bellman_retired(t,st,sol,par,retirement):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,st,poc:,0,retirement[2]] # leave/ignore points on constraint
    m = sol.m[t,st,poc:,0,retirement[2]]
    v = sol.v[t,st,poc:,0,retirement[2]]
    q = sol.q[t,st,:,0,retirement[2]]
    v_plus_raw = sol.v_plus_raw[t,st,:,0,retirement[2]]
    pi = transitions.survival(t,st,par)     

    # a. solution
    c[:] = utility.inv_marg_func(q,par)
    m[:] = par.grid_a + c
    v[:] = utility.func(c,0,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*par.grid_a)   

    # b. add points on constraint
    points_on_constraint(t,st,0,sol,par,retirement)    

@njit(parallel=True)
def solve_bellman_work(t,st,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,st,poc:,1,0] # ignore/leave points on constraint
    m = sol.m[t,st,poc:,1,0]
    v = sol.v[t,st,poc:,1,0]
    q = sol.q[t,st,:,1,0]
    v_plus_raw = sol.v_plus_raw[t,st,:,1,0]
    pi = transitions.survival(t,st,par)     

    # a. raw solution
    c_raw = utility.inv_marg_func(q,par)
    m_raw = par.grid_a + c_raw
    v_raw = utility.func(c_raw,1,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*par.grid_a)

    # b. re-interpolate to common grid
    idx = np.argsort(m_raw)
    m[:] = m_raw[idx]

    if is_sorted(idx): # no need for upper envelope
        c[:] = c_raw
        v[:] = v_raw
    else:
        print('envelope')
        envelope = upperenvelope.create(utility.func)
        envelope(par.grid_a,m_raw,c_raw,v_raw,m, # input
                 c,v, # output
                 1,par) # args for utility function

    # c. add points on constraint 
    points_on_constraint(t,st,1,sol,par,[0,0,0])
    
@njit(parallel=True)
def points_on_constraint(t,st,d,sol,par,retirement):
    """add points on the constraint"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    low_c = sol.c[t,st,poc,d,retirement[2]] # lowest point of the inner solution
    c = sol.c[t,st,:poc,d,retirement[2]] # only consider points on constraint
    m = sol.m[t,st,:poc,d,retirement[2]]
    v = sol.v[t,st,:poc,d,retirement[2]]

    # add points on constraint
    if low_c > 1e-6:
        c[:] = np.linspace(1e-6,low_c-1e-6,poc)
    else:
        c[:] = np.linspace(low_c/3,low_c/2,poc)
    m[:] = c[:]
    
    if d == 0:
        v[:] = post_decision.value_of_choice_retired(t,st,m,c,sol,par,retirement)
    else:
        v[:] = post_decision.value_of_choice_work(t,st,m,c,sol,par)


@njit(parallel=True)
def post_and_egm(t,st,sol,par,retirement):
    """run both post decision and egm for retired and working"""

    post_decision.compute_retired(t,st,sol,par,retirement)
    solve_bellman_retired(t,st,sol,par,retirement)  
    post_decision.compute_work(t,st,sol,par)                    
    solve_bellman_work(t,st,sol,par)   

@njit(parallel=True)
def recalculate(t,st,sol,par):
    """recalculate solution to keep track of eligibility of erp and two year rule, 
       which depends on when one retires"""

    # in order to keep track of eligibility of erp and two year rule we recalculate erp in the relevant years
    # remember the three options are:
    # 1. erp with two year rule if retirement_age >= 62
    # 2. erp with no two year rule if 60 <= retirement_age <= 61
    # 3. no erp if retirement_age < 60

    # This is done with the "retirement lists"
    # 1. element is where to get the t+1 solution from
    # 2. element is which erp system is in action
    # 3. element is where to store the t solution
    # 0 is the "main solution", 1 and 2 are "ad hoc/extra solutions". 1 is erp with no two year rule, 2 is no erp       

    # This is the first period, where the pension payment can differ depending on the retirement age
    # If one retires now one gets erp with two year rule, so this is the main solution
    # But we also need to store a solution, where agents get erp without two year rule and no erp at all
    # These solutions are then used later on                     
    if transitions.age(t+1) == 64:
        retirement = [[0,0,0],[0,1,1],[0,2,2]] # Calculates 3 solutions: full erp, erp without two year rule, no erp
        for ir in range(len(retirement)):
            post_and_egm(t,st,sol,par,retirement[ir])                                                                     

    elif 62 <= transitions.age(t+1) <= 63: 
        retirement = [[0,0,0],[1,1,1],[2,2,2]]
        for ir in range(len(retirement)):   
            post_and_egm(t,st,sol,par,retirement[ir])
                                                    
    # Now we jump, so if agents retire now they don't satisfy the two year rule
    elif transitions.age(t+1) == 61:
        retirement = [[1,1,0],[2,2,2]] # Calculates 2 solutions, since full erp is no longer relevant
        for ir in range(len(retirement)):   
            post_and_egm(t,st,sol,par,retirement[ir])

    elif transitions.age(t+1) == 60:
        retirement = [[0,1,0],[2,2,2]]
        for ir in range(len(retirement)):  
            post_and_egm(t,st,sol,par,retirement[ir])

    # Now we jump, so if agent retire now they don't receive erp
    elif transitions.age(t+1) == 59: 
        post_and_egm(t,st,sol,par,[2,2,0]) # Only 1 relevant solution