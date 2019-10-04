# -*- coding: utf-8 -*-
"""RetirementModel

Solves the retirement model from the "Dynamic Programming" course.

"""

##############
# 1. imports #
##############

import yaml
yaml.warnings({'YAMLLoadWarning': False})

import time
import numpy as np
from numba import boolean, int32, double, njit, prange
from numba.typed import Dict
import itertools

# consav package
from consav import linear_interp # for linear interpolation
from consav import misc # various tools
from consav import ModelClass # baseline model class

# local modules
import utility
import last_period
import post_decision
import egm
import simulate
import figs
import funs
import transitions

############
# 2. model #
############

class RetirementModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def __init__(self,couples=False,name='baseline',load=False,solmethod='egm',compiler='vs',**kwargs):
        """ basic setup

        Args:

            name (str,optional): name, used when saving/loading
            load (bool,optinal): load from disc
            solmethod (str,optional): solmethod, used when solving
            compiler (str,optional): compiler, 'vs' or 'intel' (used for C++)
             **kwargs: change to baseline parameter in .par
            
        Define parlist, sollist and simlist contain information on the
        model parameters and the variables when solving and simulating.

        Call .setup(**kwargs).

        """        

        self.name = name 
        self.solmethod = solmethod
        self.compiler = compiler
        self.vs_path = 'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
        self.intel_path = 'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/'
        self.intel_vs_version = 'vs2017'
        self.couples = couples  # type of model
        self.model_run = 0      # if couple model, we have to solve model 2 times (first single then couple)
        self.prep = True

        # a. define subclasses
        parlist = [ # (name,numba type), parameters, grids etc.
            
            # boolean
            ('couples',boolean),

            # time parameters
            ('start_T',int32),
            ('end_T',int32),
            ('forced_T',int32),
            ('T',int32),
            ('Tr',int32),
            ('dummy_t',int32),            

            # savings
            ('R',double),

            # grids            
            ('grid_a',double[:]),     
            ('a_max',int32), 
            ('a_phi',int32),
            ('Na',int32),
            ('poc',int32), 
            ('Nxi',int32), 

            # tax system
            ('tau_upper',double),
            ('tau_LMC',double),
            ('WD',double),
            ('WD_upper',double),
            ('tau_c',double),
            ('y_low',double),
            ('y_low_m',double),
            ('y_low_u',double),
            ('tau_h',double),
            ('tau_l',double),
            ('tau_m',double),
            ('tau_u',double),
            ('tau_max',double),    

            # retirement system
            ('ret_system',Dict),
            ('oap_base',double),
            ('oap_add',double),
            ('erp_high',double),

            # misc
            ('tol',double),
            ('do_print',boolean),
            ('do_simple_w',boolean),
            ('cppthreads',int32),   

            # simulation
            ('simT',int32), 
            ('simN',int32), 
            ('sim_seed',int32),  
            ('simM_init',double[:]),
            ('simStates',double[:]), 

            # states
            ('d_choice',int32[:]),
            ('age_dif',int32[:]),  
            ('states',int32[:,:]),
            ('var',Dict),   
            ('labor_inc_array',double[:,:]),
            ('survival_array',double[:,:]),          

            # preference parameters
            ('rho',double), 
            ('beta',double),
            ('alpha_0_male',double), 
            ('alpha_0_female',double),
            ('alpha_1',double),
            ('gamma',double),
                
            # uncertainty/variance parameters
            ('sigma_eta',double), 
            ('sigma_xi_men',double), 
            ('sigma_xi_women',double),

            # initial estimations
            ('reg_labor_male',Dict),
            ('reg_labor_female',Dict),
            ('reg_survival_male',Dict),
            ('reg_survival_female',Dict),
            ('reg_pension_male',Dict),
            ('reg_pension_female',Dict)                             
        ]

        if self.couples:
            dummy_list = [
                ('pareto_w',double),
                ('v',double),
                ('phi_0_male',double),
                ('phi_0_female',double),
                ('phi_1',double),
                ('sigma_xi_corr',double)
            ]
        else:
            dummy_list = []

        parlist = parlist + dummy_list
        
        sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:]),
            ('m',double[:,:,:,:,:]),
            ('v',double[:,:,:,:,:])                         
        ]        

        simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:]),

            # dummies and probabilities
            ('alive',double[:,:]), 
            ('probs',double[:,:]), 
            ('ret_age',double[:]),                      
            ('euler',double[:,:]),

            # random shocks
            ('unif',double[:,:]),
            ('deadP',double[:,:]),
            ('inc_shock',double[:,:]),

            # states
            ('states',double[:])
        ] 


        # b. create subclasses
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        # note: the above returned classes are in a format where they can be used in numba functions

        # c. load
        if load:
            self.load()
        else:
            self.setup(**kwargs)

    def setup(self,**kwargs):
        """ define baseline values and update with user choices

        Args:

             **kwargs: change to baseline parameters in .par

        """     
        # parameters identical for single and couple model
        # boolean
        self.par.couples = self.couples

        # time parameters
        self.par.start_T = 57   # start age
        self.par.end_T = 110    # end age
        self.par.forced_T = 77  # forced retirement age

        # savings
        self.par.R = 1.03       # interest rate             

        # grids
        self.par.a_max = 100    # 10 mio. kr. denominated in 100.000 kr
        self.par.a_phi = 1.1    # curvature of grid
        self.par.Na = 150       # total no. of points in a_grid=Na+poc
        self.par.poc = 20       # no. of points on constraint
        self.par.Nxi = 8        # no. of GH-points
        
        # tax system
        self.par.tau_upper = 0.59
        self.par.tau_LMC = 0.08
        self.par.WD = 0.4
        self.par.WD_upper = 12300/100000
        self.par.tau_c = 0.2554
        self.par.y_low = 41000/100000
        self.par.y_low_m = 279800/100000
        self.par.y_low_u = 335800/100000
        self.par.tau_h = 0.08
        self.par.tau_l = 0.0548
        self.par.tau_m = 0.06
        self.par.tau_u = 0.15

        # retirement system
        self.par.ret_system = {
            64: [[0,0,0],[0,1,1],[0,2,2]],    # the list indicates how to solve the model in the given time period
            63: [[0,0,0],[1,1,1],[2,2,2]],    # 1. element is where to extract t+1 solution
            62: [[0,0,0],[1,1,1],[2,2,2]],    # 2. element is retirement age/status
            61: [[1,1,0],[2,2,2]],            # 3. element is, where to store solution t
            60: [[0,1,0],[2,2,2]],
            59: [[2,2,0]]}        
        self.par.oap_base = 61152   # base rate
        self.par.oap_add = 61560    # tillæg
        self.par.erp_high = 182780  # erp with two year rule

        # misc
        self.par.tol = 1e-6
        self.par.do_print = True
        self.par.do_simple_w = False
        self.par.cppthreads = 1   

        # simulation
        self.par.simN = 1000
        self.par.sim_seed = 1998

        # parameters different for single and couple model 
        if self.couples:      
            # choices
            self.par.d_choice = np.arange(0,4) # same as [0,1,2,3]
            self.par.age_dif = np.arange(-4,5) # same as [-4,-3,-2,-1,0,1,2,3,4]

            # states
            self.par.states = np.array(list(itertools.product([0, 1], repeat=4)))                                 # 4 dummy states = 16 combinations
            self.par.var = {'female_elig': 0, 'female_high_skilled': 1, 'male_elig': 2, 'male_high_skilled': 3}   # keep track of placement                        

            # preference parameters
            self.par.rho = 0.96                         # crra
            self.par.beta = 0.98                        # time preference
            self.par.pareto_w = 0.5                     # pareto weight 
            self.par.v = 0.048                          # equivalence scale            
            self.par.alpha_0_male = 0.160               # constant
            self.par.alpha_0_female = 0.119             # constant
            self.par.alpha_1 = 0.053                    # high skilled
            self.par.phi_0_male = 1.187                 # constant, joint leisure
            self.par.phi_0_female = 1.671               # constant, joint leisure
            self.par.phi_1 = -0.621                     # high skilled, joint leisure            
            self.par.gamma = 0.08                       # bequest motive

            # uncertainty/variance parameters
            self.par.sigma_eta = 0.435                  # taste shock
            self.par.sigma_xi_men = np.sqrt(0.544)      # income shock
            self.par.sigma_xi_women = np.sqrt(0.399)    # income shock          
            self.par.sigma_xi_corr = 0                  # correlation of income shocks (couples)

            # initial estimations
            self.par.reg_labor_male = {'cons': -5.999, 'high_skilled': 0.262, 'age': 0.629, 'age2': -0.532, 'children': 0.060}
            self.par.reg_labor_female = {'cons': -4.002, 'high_skilled': 0.318, 'age': 0.544, 'age2': -0.453, 'children': -0.018}            
            self.par.reg_survival_male = {'cons': -10.338, 'age': 0.097}
            self.par.reg_survival_female = {'cons': -11.142, 'age': 0.103}
            self.par.reg_pension_male = {'cons': -41.161, 'age': 0.072, 'age2': -0.068, 'high_skilled': 0.069, 'children': 0.026, 'log_wealth': 8.864, 'log_wealth2': -0.655, 'log_wealth3': 0.016}
            self.par.reg_pension_female = {'cons': -19.000, 'age': 0.039, 'age2': -0.037, 'high_skilled': 0.131, 'children': -0.024, 'log_wealth': 4.290, 'log_wealth2': -0.327, 'log_wealth3': 0.008}            

        else:
            # choices
            self.par.d_choice = np.arange(0,2)  # same as [0,1]
            self.par.age_dif = np.arange(1)     # same as [0]

            # states
            self.par.states = list(itertools.product([0, 1], repeat=4))             # 4 dummy states = 16 combinations
            self.par.var = {'male': 0, 'elig': 1, 'high_skilled': 2, 'children': 3} # keep track of placement            

            # preference parameters
            self.par.rho = 0.96                         # crra
            self.par.beta = 0.98                        # time preference
            self.par.alpha_0_male = 0.160               # constant, own leisure
            self.par.alpha_0_female = 0.119 + 0.05      # constant, own leisure
            self.par.alpha_1 = 0.053                    # high skilled, own leisure
            self.par.alpha_2 = -0.036                   # children, own leisure
            self.par.gamma = 0.08                       # bequest motive

            # uncertainty/variance parameters
            self.par.sigma_eta = 0.435                  # taste shock
            self.par.sigma_xi_men = np.sqrt(0.544)      # income shock
            self.par.sigma_xi_women = np.sqrt(0.399)    # income shock          

            # initial estimations
            self.par.reg_labor_male = {'cons': -15.956, 'high_skilled': 0.230, 'age': 0.934, 'age2': -0.770, 'children': 0.151}
            self.par.reg_labor_female = {'cons': -18.937, 'high_skilled': 0.248, 'age': 1.036, 'age2': -0.856, 'children': 0.021}            
            self.par.reg_survival_male = {'cons': -10.338, 'age': 0.097}
            self.par.reg_survival_female = {'cons': -11.142, 'age': 0.103}
            self.par.reg_pension_male = {'cons': -57.670, 'age': 0.216, 'age2': -0.187, 'high_skilled': 0.142, 'children': 0.019, 'log_wealth': 12.057, 'log_wealth2': -0.920, 'log_wealth3': 0.023}
            self.par.reg_pension_female = {'cons': -47.565, 'age': 0.098, 'age2': -0.091, 'high_skilled': 0.185, 'children': -0.032, 'log_wealth': 10.062, 'log_wealth2': -0.732, 'log_wealth3': 0.018}

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val

        # c. create/update parameters, which depends on other parameters
        self.par.T = self.par.end_T - self.par.start_T + 1
        self.par.Tr = self.par.forced_T - self.par.start_T + 1   
        self.par.dummy_t = transitions.inv_age(min(self.par.ret_system),self.par)    
        self.par.simT = self.par.T            
        self.par.tau_max = self.par.tau_l + self.par.tau_m + self.par.tau_u + self.par.tau_c + self.par.tau_h - self.par.tau_upper            
        self.par.simM_init = 5*np.ones(self.par.simN)
        self.par.simStates = np.zeros(self.par.simN,dtype=int)           

        # d. precompute state variables from initial estimations (labor income and survival probs)
        self.par.labor_inc_array = transitions.labor_income_fill_out(self.par)
        self.par.survival_array = transitions.survival_fill_out(self.par)

        # e. setup_grids
        self.setup_grids()
        
    def setup_grids(self):
        """ construct grids for states and shocks """

        # a. post-decision states (unequally spaced vector of length Na)
        self.par.grid_a = misc.nonlinspace(1e-6,self.par.a_max,self.par.Na,self.par.a_phi)
        
        # b. shocks (quadrature nodes and weights using GaussHermite)
        self.par.xi_men,self.par.xi_men_w = funs.GaussHermite_lognorm(self.par.sigma_xi_men,self.par.Nxi)
        self.par.xi_women,self.par.xi_women_w = funs.GaussHermite_lognorm(self.par.sigma_xi_women,self.par.Nxi)        


    #########
    # solve #
    #########

    def _solve_prep(self):
        """ allocate memory for solution """

        # prep
        par = self.par
        num_st = len(par.states)            # number of states
        num_grid = par.Na + par.poc         # number of points in grid
        num_d = len(par.d_choice)           # number of choices        
        num_recalc = len(par.ret_system)    # number of periods to recalculate solution/erp

        if self.model_run == 0: # single model
            
            # solution
            self.sol.c = np.nan*np.ones((par.T,num_st,1,num_grid,num_d))   
            self.sol.m = np.nan*np.zeros((par.T,num_st,1,num_grid,num_d))
            self.sol.v = np.nan*np.zeros((par.T,num_st,1,num_grid,num_d))   
            self.sol.c_dummy = np.nan*np.zeros((num_recalc,num_grid,num_d,2))        
            self.sol.m_dummy = np.nan*np.zeros((num_recalc,num_grid,num_d,2))
            self.sol.v_dummy = np.nan*np.zeros((num_recalc,num_grid,num_d,2))

            # interpolation
            self.sol.c_plus_interp = np.nan*np.zeros((num_grid,num_d))
            self.sol.v_plus_interp = np.nan*np.zeros((num_grid,num_d)) 

            # post decision - only need the inner points of the grid
            self.sol.q = np.nan*np.zeros((par.Na,num_d))
            self.sol.v_plus_raw = np.nan*np.zeros((par.Na,num_d))

        else: # couple model

            # prep
            num_ad = len(par.age_dif)   # number of age differences

            # save single solution
            self.sol.c_singles = self.sol.c[:]
            self.sol.m_singles = self.sol.m[:]
            self.sol.v_singles = self.sol.v[:]

            # solution
            self.sol.c = np.nan*np.ones((par.T,num_st,num_ad,num_grid,num_d))   
            self.sol.m = np.nan*np.zeros((par.T,num_st,num_ad,num_grid,num_d))
            self.sol.v = np.nan*np.zeros((par.T,num_st,num_ad,num_grid,num_d))   
            self.sol.c_dummy = np.nan*np.zeros((num_recalc,num_grid,2))        
            self.sol.m_dummy = np.nan*np.zeros((num_recalc,num_grid,2))
            self.sol.v_dummy = np.nan*np.zeros((num_recalc,num_grid,2))

            # interpolation - reuse
            self.sol.c_plus_interp[:,:] = np.nan
            self.sol.v_plus_interp[:,:] = np.nan            

            # post decision - reuse
            self.sol.q[:,:] = np.nan
            self.sol.v_plus_raw[:,:] = np.nan


    @njit(parallel=True)
    def solve(self):
        """ solve the model """
        self.solve_singles()
        if self.couples:
            self.model_run = 1 # tell to run the couple solution now
            self.solve_couples()


    @njit(parallel=True)
    def solve_singles(self):
        """ solve the model for singles """

        # prep
        par = self.par
        sol = self.sol

        # a. allocate solution
        if self.prep:
            self._solve_prep()
        
        # b. backwards induction
        for ad in par.age_dif:                  # loop over age differences for couples
            for st in prange(len(par.states)):  # loop over states (solve model many times)
                for t in range(par.T-1,-1,-1):  # same as reversed(range(par.T))
                
                    # i. last period
                    if t == par.T-1:
                    
                        last_period.solve(t,st,ad,sol,par)

                    # ii. if forced to retire (retirement decision is made one period ahead)
                    elif t+1 >= par.Tr-1:

                        post_decision.compute_singles(t,st,ad,[0],sol,par)
                        egm.solve_bellman_singles(t,st,ad,[0],sol,par)
                    
                    # iii. oap period (retirement decision is made one period ahead)
                    elif transitions.age(t+1,par) >= max(par.ret_system)+1:
                        
                        egm.solve(t,st,ad,sol,par) # just a wrapper, which runs both post decision and egm functions

                    # iv. erp period - here we have to recalculate solutions if eligible to erp
                    elif (min(par.ret_system) <= transitions.age(t+1,par) <= max(par.ret_system)) and (transitions.state_translate(st,'elig',par) == 1):
                        
                        for ir in par.ret_system[transitions.age(t+1,par)]:
                            egm.solve(t,st,ad,sol,par,ir)                    

                    # v. before erp periods
                    else:
                        
                        egm.solve(t,st,ad,sol,par,[0,2,0]) # take t+1 sol from main sol, assume no erp, store t sol in main sol                    


    @njit(parallel=True) # have not beed used yet
    def solve_couples(self):
        """ solve the model for couples """

        # prep
        par = self.par
        sol = self.sol

        # a. allocate solution
        self._solve_prep()
        
        # b. backwards induction
        for ad in par.age_dif:                  # loop over age differences for couples
            for st in prange(len(par.states)):  # loop over states (solve model many times)
                for t in range(par.T-1,-1,-1):  # same as reversed(range(par.T))
                
                    # i. last period
                    if t == par.T-1:
                        last_period.solve(t,st,ad,sol,par)

                    # ii. if both are forced to retire (retirement decision is made one period ahead)
                    elif t+1 >= par.Tr-1 and t+1+ad >= par.Tr-1: # ad can be both negative and positive
                        d = [0] # possible choices

                    # iii. if husband is forced to retire, but not wife
                    elif t+1 >= par.Tr-1 and t+1+ad < par.Tr-1:
                        d = [0,1]   # possible choices

                    # iv. if wife is forced to retire but not husband
                    elif t+1 < par.Tr-1 and t+1+ad >= par.Tr-1:
                        d = [0,2]   # possible choices

                    # v. oap periods for both
                    elif max(transitions.age(t+1),transitions.age(t+1)+ad) >= max(par.ret_system)+1:
                        retirement = [0,0,0] # no need to recalculate solution

                    elif min(par):
                        pass

                    # vi. oap periods for 

                    # v. oap period (retirement decision is made one period ahead)
                    elif transitions.age(t+1) >= max(par.ret_system['jumps']):
                        egm.solve(t,st,ad,sol,par) # just a wrapper, which runs both post decision and egm functions

                    # iv. erp period - here we have to recalculate solutions if eligible to erp
                    elif (min(par.ret_system['jumps'])-1 <= transitions.age(t+1) <= max(par.ret_system['jumps'])-1) and (transitions.state_translate(st,'elig',par) == 1):
                        for ir in par.ret_system[transitions.age(t+1)]:
                            egm.solve(t,st,ad,sol,par,ir)                    

                    # v. before erp periods
                    else:
                        egm.solve(t,st,ad,sol,par,[0,2,0]) # take t+1 sol from main sol, assume no erp, store t sol in main sol                    


    ############
    # simulate #
    ############

    def _simulate_prep(self):
        """ allocate memory for simulation and draw random numbers """

        # set seed
        np.random.seed(self.par.sim_seed)

        # prep
        par = self.par
        sim = self.sim

        # solution
        sim.c = np.nan*np.zeros((par.simT,par.simN))
        sim.m = np.nan*np.zeros((par.simT,par.simN))
        sim.a = np.nan*np.zeros((par.simT,par.simN))
        sim.d = np.nan*np.zeros((par.simT,par.simN)) # retirement choice

        # dummies and probabilities
        sim.alive = np.ones((par.simT,par.simN)) #dummy for alive
        sim.probs = np.zeros((par.simT,par.simN)) # retirement probs
        sim.ret_age = np.nan*np.zeros(par.simN) # retirement age

        # interpolation
        sim.c_interp = np.nan*np.zeros((par.simT,par.simN,2))
        sim.v_interp = np.nan*np.zeros((par.simT,par.simN,2)) 
        sim.euler = np.nan*np.zeros((par.simT-1,par.simN))

        # b. initialize m and d
        sim.m[0,:] = par.simM_init        
        sim.d[0,:] = np.ones(par.simN) # all is working at t=0

        # c. states
        sim.states = par.simStates

        # d. draw random shocks
        sim.unif = np.random.rand(par.simT,par.simN) # taste shocks
        sim.deadP = np.random.rand(par.simT,par.simN) # death probs
        sim.inc_shock = np.nan*np.zeros((par.Tr-1,par.simN)) # income shocks
        for i in range(len(sim.states)): 
            if transitions.state_translate(sim.states[i],'male',par) == 1:
                sim.inc_shock[:,i] = np.random.lognormal(-0.5*(par.sigma_xi_men**2),par.sigma_xi_men,size=par.Tr-1)
            else:
                sim.inc_shock[:,i] = np.random.lognormal(-0.5*(par.sigma_xi_women**2),par.sigma_xi_women,size=par.Tr-1)


    def simulate(self,accuracy_test=False):
        """ simulate model """

        # a. allocate memory and draw random numbers
        if self.prep:
            self._simulate_prep()

        # b. simulate
        self.par.simT = self.par.T
        simulate.lifecycle(self.sim,self.sol,self.par,accuracy_test)


    def test(self):
        """ method for specifying test """
        
        # a. save print status
        do_print = self.par.do_print
        self.par.do_print = False

        # b. test run
        self.solve()

        # c. timed run
        tic = time.time()  
        self.solve()
        toc = time.time()
        print(f'solution time: {toc-tic:.1f} secs')

        # d. reset print status
        self.par.do_print = do_print        


#to debug code
from consav import runtools
runtools.write_numba_config(disable=1,threads=8)
model = RetirementModelClass(name='baseline',solmethod='egm')
model.solve()
model.simulate(accuracy_test=True)
