# global modules
from numba import njit, prange
import numpy as np

# consav
from consav import linear_interp
from consav import upperenvelope 

# local modules
import utility
import post_decision
import transitions


@njit(parallel=True)
def is_sorted(a):
    """ True if a is sorted, otherwise False"""
    for i in range(a.size-1):
         if a[i+1] < a[i]:
               return False
    return True

@njit(parallel=True)
def solve_bellman_singles(t,st,ad,d,sol,par,retirement):
    """ solve the bellman equation for singles using the endogenous grid method
    
        Args:
        
            t=time, st=state, ad=age difference
            d=retirement choice
            sol=solution, par=parameters
            retirement=list with info about erp status"""

    # unpack solution
    poc = par.poc # points on constraint
    if retirement[2] == 0:
        c = sol.c[t,st,ad,poc:] # leave/ignore points on constraint
        m = sol.m[t,st,ad,poc:]
        v = sol.v[t,st,ad,poc:]
    else:
        c = sol.c_dummy[t-par.dummy_t,poc:,:,retirement[2]-1] 
        m = sol.m_dummy[t-par.dummy_t,poc:,:,retirement[2]-1]
        v = sol.v_dummy[t-par.dummy_t,poc:,:,retirement[2]-1]        

    # unpack rest
    q = sol.q[:,:]
    v_plus_raw = sol.v_plus_raw[:,:]        
    pi = transitions.survival_look_up(t,st,par)   

    # loop over the choices
    for id in d:

        # a. raw solution
        c[:,id] = utility.inv_marg_func(q[:,id],par)
        m[:,id] = par.grid_a + c[:,id]
        v[:,id] = utility.func(c[:,id],id,st,par) + par.beta*(pi*v_plus_raw[:,id] + (1-pi)*par.gamma*par.grid_a)          

        # b. reset c,m,v pointers
        if retirement[0] != 0:
            c = sol.c[t,st,ad,poc:]
            m = sol.m[t,st,ad,poc:]
            v = sol.v[t,st,ad,poc:]  

        # c. upper envelope
        if id == 1:

            # prep
            idx = np.argsort(m[:,id]) # this is our m_raw
            
            #1. check if upper envelope is necessary
            if is_sorted(idx): # no need for upper envelope
                pass

        #     # 2. upper envelope
        #     else: # upper envelope

        #         # define raw
        #         # m_raw = m[:,id]
        #         # c_raw = c[:,id]
        #         # v_raw = v[:,id]
        #         # m[:,id] = m_raw[idx]
                
        #         # run upper envelope
        #         # envelope = upperenvelope.create(utility.func)
        #         # envelope(par.grid_a,m_raw,c_raw,v_raw,m[:,id],  # input
        #         #         c[:,id],v[:,id],                        # output
        #         #         id,st,par)                              # args for utility function 

        # d. add points on constraint
        points_on_constraint(t,st,ad,id,sol,par,retirement)


@njit(parallel=True)
def points_on_constraint(t,st,ad,d,sol,par,retirement):
    """ put points on constraint
    
        Args:
        
            t=time, st=state, ad=age difference
            d=retirement choice
            sol=solution, par=parameters
            retirement=list with info about erp status"""    

    # unpack (helps numba optimize)
    tol = par.tol # tolerance level
    poc = par.poc # points on constraint
    v_plus_interp = sol.v_plus_interp[:poc,:]    
    if retirement[2] == 0 or d == 1:
        low_c = sol.c[t,st,ad,poc,d]    # lowest point of the inner solution
        c = sol.c[t,st,ad,:poc,d]       # only consider points on constraint
        m = sol.m[t,st,ad,:poc,d]
        v = sol.v[t,st,ad,:poc,d]
    else:
        low_c = sol.c_dummy[t-par.dummy_t,poc,d,retirement[2]-1] 
        c = sol.c_dummy[t-par.dummy_t,:poc,d,retirement[2]-1] 
        m = sol.m_dummy[t-par.dummy_t,:poc,d,retirement[2]-1]
        v = sol.v_dummy[t-par.dummy_t,:poc,d,retirement[2]-1]      

    # add points on constraint
    if low_c > tol:
        c[:] = np.linspace(tol,low_c-tol,poc)
    else: # a small fix if low_c is lower than tol
        c[:] = np.linspace(low_c/3,low_c/2,poc)
    m[:] = c[:]

    # compute value-of-choice
    post_decision.value_of_choice(t,st,ad,d,m,c,v,v_plus_interp,sol,par,retirement)


@njit(parallel=True)
def solve_singles(t,st,ad,d,sol,par,retirement=[0,0,0]):
    """ wrapper which calls both post_decision.compute_singles and egm.solve_bellman_singles"""

    post_decision.compute_singles(t,st,ad,d,sol,par,retirement)
    solve_bellman_singles(t,st,ad,d,sol,par,retirement)  

















@njit(parallel=True)
def solve_bellman_retired(t,st,ad,sol,par,retirement=[0,0,0]):
    """ Solve the bellman equation if retired using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    if retirement[2] == 0:
        c = sol.c[t,st,ad,poc:,0] # leave/ignore points on constraint
        m = sol.m[t,st,ad,poc:,0]
        v = sol.v[t,st,ad,poc:,0]
    else:
        c = sol.c_dummy[t-par.dummy_t,poc:,0,retirement[2]-1] 
        m = sol.m_dummy[t-par.dummy_t,poc:,0,retirement[2]-1]
        v = sol.v_dummy[t-par.dummy_t,poc:,0,retirement[2]-1]
        
    q = sol.q[:,0]
    v_plus_raw = sol.v_plus_raw[:,0]        
    pi = transitions.survival_look_up(t,st,par)     

    # a. solution
    c[:] = utility.inv_marg_func(q,par)
    m[:] = par.grid_a + c
    v[:] = utility.func(c,0,st,par) + par.beta*(pi*v_plus_raw + (1-pi)*par.gamma*par.grid_a)   

    # b. add points on constraint
    points_on_constraint(t,st,ad,0,sol,par,retirement)    

@njit(parallel=True)
def solve_bellman_work(t,st,ad,sol,par):
    """ Solve the bellman equation if working using the endogenous grid method"""

    # unpack (helps numba optimize)
    poc = par.poc # points on constraint
    c = sol.c[t,st,ad,poc:,1] # ignore/leave points on constraint
    m = sol.m[t,st,ad,poc:,1]
    v = sol.v[t,st,ad,poc:,1]
    q = sol.q[:,1]
    v_plus_raw = sol.v_plus_raw[:,1]
    pi = transitions.survival_look_up(t,st,par)     

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
        print('envelope',t,st)
        envelope = upperenvelope.create(utility.func)
        envelope(par.grid_a,m_raw,c_raw,v_raw,m, # input
                 c,v, # output
                 1,par) # args for utility function

    # c. add points on constraint 
    points_on_constraint(t,st,ad,1,sol,par)