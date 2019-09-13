# -*- coding: utf-8 -*-
"""RetirementModel

Solves the single model

"""

##############
# 1. imports #
##############

import yaml
yaml.warnings({'YAMLLoadWarning': False})

import time
import numpy as np
from numba import boolean, int32, int64, float64, double, njit, prange, typeof
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
import solution

############
# 2. model #
############

class SingleClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def __init__(self,name='baseline_singles',load=False,compiler='vs',**kwargs):
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
        self.compiler = compiler
        self.vs_path = 'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
        self.intel_path = 'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/'
        self.intel_vs_version = 'vs2017'

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
            ('a_phi',double),
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

            # retirement
            ('oap_age',int32),
            ('two_year',int32),            
            ('erp_age',int32),
            ('len_ret',int32),
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
            ('simStates',int32[:]), 

            # states
            ('d_choice',int32[:]),
            ('age_dif',int32[:]),  
            ('states',int32[:,:]),   
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
            ('xi_men',double[:]),
            ('xi_men_w',double[:]),
            ('xi_women',double[:]),
            ('xi_women_w',double[:])                                      
        ]
        
        sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:]),
            ('m',double[:,:,:,:,:]),
            ('v',double[:,:,:,:,:]),

            ('c_dummy',double[:,:,:,:]),
            ('m_dummy',double[:,:,:,:]),
            ('v_dummy',double[:,:,:,:]),                                 

            ('c_plus_interp',double[:,:]),
            ('v_plus_interp',double[:,:]),

            ('q',double[:,:]),
            ('v_plus_raw',double[:,:])                            
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
            ('states',int32[:])
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
        # boolean
        self.par.couples = False

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
        self.par.poc = 10       # no. of points on constraint
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

        # retirement
        self.par.oap_age = 65
        self.par.two_year = 62
        self.par.erp_age = 60
        self.par.len_ret = 6
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

        # choices
        self.par.d_choice = np.arange(0,2)  # same as [0,1]
        self.par.age_dif = np.arange(1)     # same as [0]

        # states
        self.par.states = np.array(list(itertools.product([0, 1], repeat=3))) # 3 dummy states = 8 combinations
        #self.par.var = {'male': 0, 'elig': 1, 'high_skilled': 2}    # keep track of placement            

        # preference parameters
        self.par.rho = 0.96                         # crra
        self.par.beta = 0.98                        # time preference
        self.par.alpha_0_male = 0.160               # constant, own leisure
        self.par.alpha_0_female = 0.119 + 0.05      # constant, own leisure
        self.par.alpha_1 = 0.053                    # high skilled, own leisure
        self.par.gamma = 0.08                       # bequest motive

        # uncertainty/variance parameters
        self.par.sigma_eta = 0.435                  # taste shock
        self.par.sigma_xi_men = np.sqrt(0.544)      # income shock
        self.par.sigma_xi_women = np.sqrt(0.399)    # income shock          

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val

        # c. create/update parameters, which depends on other parameters
        self.par.T = self.par.end_T - self.par.start_T + 1
        self.par.Tr = self.par.forced_T - self.par.start_T + 1   
        self.par.dummy_t = transitions.inv_age(self.par.erp_age-1,self.par)    
        self.par.simT = self.par.T            
        self.par.tau_max = self.par.tau_l + self.par.tau_m + self.par.tau_u + self.par.tau_c + self.par.tau_h - self.par.tau_upper            
        self.par.simM_init = 5*np.ones(self.par.simN)
        self.par.simStates = np.random.randint(len(self.par.states),size=self.par.simN)#np.zeros(self.par.simN,dtype=int)   

        # d. precompute state variables from initial estimations (labor income and survival probs)
        self.par.labor_inc_array = transitions.labor_income_fill_out(self.par)
        self.par.survival_array = transitions.survival_fill_out(self.par)                    


    def setup_grids(self):
        """ construct grids for states and shocks """

        # a. post-decision states (unequally spaced vector of length Na)
        self.par.grid_a = misc.nonlinspace(self.par.tol,self.par.a_max,self.par.Na,self.par.a_phi)
        
        # b. shocks (quadrature nodes and weights using GaussHermite)
        self.par.xi_men,self.par.xi_men_w = funs.GaussHermite_lognorm(self.par.sigma_xi_men,self.par.Nxi)
        self.par.xi_women,self.par.xi_women_w = funs.GaussHermite_lognorm(self.par.sigma_xi_women,self.par.Nxi)        


    #########
    # solve #
    #########
    def _solve_prep(self):
        """ allocate memory for solution """

        # setup_grids
        self.setup_grids()        

        # prep
        num_st = len(self.par.states)            # number of states
        num_grid = self.par.Na + self.par.poc    # number of points in grid
        num_d = len(self.par.d_choice)           # number of choices        
        num_recalc = self.par.len_ret            # number of periods to recalculate solution/erp
            
        # solution
        self.sol.c = np.nan*np.ones((self.par.T,num_st,1,num_grid,num_d))   
        self.sol.m = np.nan*np.zeros((self.par.T,num_st,1,num_grid,num_d))
        self.sol.v = np.nan*np.zeros((self.par.T,num_st,1,num_grid,num_d))   
        self.sol.c_dummy = np.nan*np.zeros((num_recalc,num_grid,num_d,2))        
        self.sol.m_dummy = np.nan*np.zeros((num_recalc,num_grid,num_d,2))
        self.sol.v_dummy = np.nan*np.zeros((num_recalc,num_grid,num_d,2))

        # interpolation
        self.sol.c_plus_interp = np.nan*np.zeros((num_grid,num_d))
        self.sol.v_plus_interp = np.nan*np.zeros((num_grid,num_d)) 

        # post decision - only need the inner points of the grid
        self.sol.q = np.nan*np.zeros((self.par.Na,num_d))
        self.sol.v_plus_raw = np.nan*np.zeros((self.par.Na,num_d))

    def solve(self):
        """ solve the model """

        # a. allocate solution
        self._solve_prep()

        # b. solve the model
        solution.solve(self.par,self.sol)
                            

    ############
    # simulate #
    ############

    def _simulate_prep(self):
        """ allocate memory for simulation and draw random numbers """

        # solution
        self.sim.c = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.m = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.a = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.d = np.nan*np.zeros((self.par.simT,self.par.simN))

        # dummies and probabilities
        self.sim.alive = np.ones((self.par.simT,self.par.simN))             # dummy for alive
        self.sim.probs = np.zeros((self.par.simT,self.par.simN))            # retirement probs
        self.sim.ret_age = np.nan*np.zeros(self.par.simN)                   # retirement age

        # initialize m and d
        self.sim.m[0,:] = self.par.simM_init        
        self.sim.d[0,:] = np.ones(self.par.simN)                            # all is working at t=0
        
        # states
        self.sim.states = self.par.simStates

        # euler errors
        self.sim.euler = np.nan*np.zeros((self.par.simT-1,self.par.simN))

        # random draws
        np.random.seed(self.par.sim_seed)        
        self.sim.unif = np.random.rand(self.par.simT,self.par.simN)         # taste shocks
        self.sim.deadP = np.random.rand(self.par.simT,self.par.simN)        # death probs
        self.sim.inc_shock = np.nan*np.zeros((self.par.Tr-1,self.par.simN)) # income shocks
        for i in range(len(self.sim.states)): 
            if transitions.state_translate(self.sim.states[i],'male',self.par) == 1:
                self.sim.inc_shock[:,i] = np.random.lognormal(-0.5*(self.par.sigma_xi_men**2),self.par.sigma_xi_men,size=self.par.Tr-1)
            else:
                self.sim.inc_shock[:,i] = np.random.lognormal(-0.5*(self.par.sigma_xi_women**2),self.par.sigma_xi_women,size=self.par.Tr-1)        


    def simulate(self,euler=False):
        """ simulate model """

        # a. allocate memory and draw random numbers 
        self._simulate_prep()
        
        # b. simulate
        simulate.lifecycle(self.sim,self.sol,self.par)

        # c. euler errors
        if euler:
            simulate.euler_error(self.sim,self.sol,self.par)


# #to debug code
# from consav import runtools
# runtools.write_numba_config(disable=1,threads=8)
# data = SingleClass(simN=2000)
# data.solve()
# data.simulate(euler=True)
