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
from numba import boolean, int32, double

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

############
# 2. model #
############

class RetirementModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def __init__(self,name='baseline',load=False,solmethod='egm',compiler='vs',**kwargs):
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
        
        # a. define subclasses
        parlist = [ # (name,numba type), parameters, grids etc.
            
            # parameters
            ('T',int32), # integer 32bit
            ('Tr',int32),
            ('rho',double), # double
            ('beta',double),
            ('alpha',double),
            ('gamma',double),
            ('sigma_eta',double),
            ('sigma_xi',double),            
            ('R',double),
            ('a_max',int32),
            ('a_phi',int32),
            ('Nxi',int32),
            ('Na',int32),
            ('poc',int32),

            # grids            
            ('grid_a',double[:]), # 1d array of doubles    
            ('xi',double[:]),        
            #('xi_w',double[:]),
            #('a_work',double[:,:]), # 2d array of doubles
            #('xi_work',double[:,:]),

            # misc
            ('tol',double),
            ('do_print',boolean), # boolean
            ('do_simple_w',boolean),
            ('cppthreads',int32),
            
            # simulation
            ('simT',int32), 
            ('simN',int32), 
            ('sim_seed',int32)
        ]
        
        sollist = [ # (name, numba type), solution data
            ('c',double[:,:,:]), # 3d array of doubles
            ('m',double[:,:,:]),
            ('v',double[:,:,:]),
            ('v_plus',double[:,:,:]),
            ('c_plus_interp',double[:,:,:]),
            ('v_plus_interp',double[:,:,:]),
            ('c_plus_retired_interp',double[:,:]), # 2d array of doubles
            ('v_plus_retired_interp',double[:,:]),
            ('q',double[:,:,:]),
            ('v_plus_raw',double[:,:,:])
        ]        

        simlist = [ # (name, numba type), simulation data
            ('p',double[:,:]),
            ('m',double[:,:]),
            ('c',double[:,:]),
            ('v',double[:,:]),
            ('a',double[:,:]),
            ('d',double[:,:]),
            ('alive',double[:,:]),
            ('c_interp',double[:,:,:]),
            ('v_interp',double[:,:,:]),        
            ('xi',double[:,:]),
            ('psi',double[:,:]),
            ('unif',double[:,:])
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

        # a. baseline parameters
        
        # demographics
        self.par.T = 110-57+1 # 57-110 years
        self.par.Tr = 77-57+1 # forced retirement at 77 years
        
        # preferences
        self.par.rho = 0.96
        self.par.beta = 0.98        
        self.par.alpha = 0.1
        self.par.gamma = 0.08

        # taste shocks
        self.par.sigma_eta = 0.435

        # income
        men_surv_to_99 = [0.99292,0.99242,0.99186,0.9905,0.98969,0.98883,0.98803,0.98674,0.98626,
        0.98516,0.98393,0.98222,0.98092,0.97806,0.97702,0.97523,0.97193,0.96871,0.96631,0.96403,0.9579,
        0.95379,0.94785,0.94092,0.9326,0.92529,0.91379,0.9071,0.89541,0.88177,0.86924,0.85226,0.83528,
        0.82236,0.79992,0.77261,0.74983,0.74447,0.72467,0.68315,0.65767,0.66008,0.60322] # DST fra tabel: HISB9, 1-dødshyppighed
        self.par.survival_probs = men_surv_to_99 + list(np.linspace(men_surv_to_99[-1],0,110-99))
        #self.par.survival_probs = [0.93332,0.92671,0.91968,0.9122,0.90353,0.89422,0.88423,0.87364,0.86206,0.85021,0.8376,0.82413,
        #0.80948,0.79404,0.77662,0.75878,0.73998,0.71921,0.69671,0.67324,0.64902,0.6217,0.59297,0.56204,
        #0.52884,0.49319,0.45634,0.417,0.37826,0.3387,0.29866,0.25961,0.22125,0.18481,0.15198,0.12157,
        #0.09393,0.07043,0.05243,0.038,0.02596,0.01707,0.01127] #DST fra tabel: HISB9
        self.par.Y = 1
        self.par.sigma_xi = 0.2
        
        # saving
        self.par.R = 1.03
        
        # grids
        self.par.a_max = 100 # denominated in 100.000 kr
        self.par.a_phi = 1.1
        self.par.Nxi = 8
        self.par.Na = 150
        self.par.poc = 10 # points on constraint
        
        # misc
        self.par.tol = 1e-8
        self.par.do_print = True
        self.par.do_simple_w = False
        self.par.cppthreads = 1

        # simulation
        self.par.simT = self.par.T
        self.par.simN = 1000
        self.par.sim_seed = 1998

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val
        
        # c. setup_grids
        self.setup_grids()
        
    def setup_grids(self):
        """ construct grids for states and shocks """

        # a. post-decision states (unequally spaced vector of length Na)
        self.par.grid_a = misc.nonlinspace(1e-6,self.par.a_max,self.par.Na,self.par.a_phi)
        
        # b. shocks (qudrature nodes and weights using GaussHermite)
        self.par.xi,self.par.xi_w = funs.GaussHermite_lognorm(self.par.sigma_xi,self.par.Nxi)

        # create tiled/broadcasted arrays to use in compute
        #self.par.a_work = np.transpose(np.array([self.par.grid_a]*self.par.Nxi))
        #self.par.xi_work = np.array([self.par.xi]*self.par.Na)

        # d. set seed
        np.random.seed(self.par.sim_seed)

    def checksum(self):
        """ print checksum """

        print(f'checksum: {np.mean(self.sol.c[0])}')

    #########
    # solve #
    #########

    def _solve_prep(self):
        """ allocate memory for solution """

        self.sol.c = np.nan*np.ones((self.par.T,self.par.Na+self.par.poc,2))        
        self.sol.m = np.nan*np.zeros((self.par.T,self.par.Na+self.par.poc,2))
        self.sol.v = np.nan*np.zeros((self.par.T,self.par.Na+self.par.poc,2))
        self.sol.v_plus = np.nan*np.zeros((self.par.T-1,self.par.Na,2))
        
        #self.sol.c_plus_interp = np.nan*np.zeros((self.par.T-1,self.par.Na*self.par.Nxi,2))
        #self.sol.v_plus_interp = np.nan*np.zeros((self.par.T-1,self.par.Na*self.par.Nxi,2))  
        self.sol.c_plus_interp = np.nan*np.zeros((self.par.T-1,self.par.Na,2))
        self.sol.v_plus_interp = np.nan*np.zeros((self.par.T-1,self.par.Na,2))  
        
        self.sol.c_plus_retired_interp = np.nan*np.zeros((self.par.T-1,self.par.Na))
        self.sol.v_plus_retired_interp = np.nan*np.zeros((self.par.T-1,self.par.Na))        
        self.sol.q = np.nan*np.zeros((self.par.T-1,self.par.Na,2))
        self.sol.v_plus_raw = np.nan*np.zeros((self.par.T-1,self.par.Na,2))

    def solve(self):
        """ solve the model """

        # a. allocate solution
        self._solve_prep()
        
        # b. backwards induction
        for t in reversed(range(self.par.T)):        
            
            # i. last period
            if t == self.par.T-1:
                
                last_period.solve(t,self.sol,self.par)

            ## ii. if forced to retire
            elif t >= self.par.Tr-1:

                post_decision.compute_retired(t,self.sol,self.par)
                egm.solve_bellman_retired(t,self.sol,self.par)
                
            # iii. all other periods
            else:
                
                post_decision.compute_retired(t,self.sol,self.par)
                post_decision.compute_work(t,self.sol,self.par)
                egm.solve_bellman_work(t,self.sol,self.par)

    ############
    # simulate #
    ############

    def _simulate_prep(self):
        """ allocate memory for simulation and draw random numbers """

        # a. allocate
        self.sim.p = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.m = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.c = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.v = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.d = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.alive = np.ones((self.par.simT,self.par.simN)) #dummy for alive
        self.sim.a = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.c_interp = np.nan*np.zeros((self.par.simT,self.par.simN,2))
        self.sim.v_interp = np.nan*np.zeros((self.par.simT,self.par.simN,2))                     

        # b. initialize m
        self.sim.m[0,:] = np.random.lognormal(np.log(25),1.2,self.par.simN) # initial m, lognormal dist
        #self.sim.m[0,:] = 10*np.ones(par.simN) # initial m        

        # c. draw random shocks
        self.sim.unif = np.random.rand(self.par.simT,self.par.simN)
        self.sim.deadP = np.random.rand(self.par.simT,self.par.simN)

    def simulate(self):
        """ simulate model """

        # a. allocate memory and draw random numbers
        self._simulate_prep()

        # b. simulate
        self.par.simT = self.par.T
        simulate.lifecycle(self.sim,self.sol,self.par)


#to debug code
from consav import runtools
runtools.write_numba_config(disable=1,threads=8)
model = RetirementModelClass(name='baseline',solmethod='egm')
model.solve()
model.simulate()
