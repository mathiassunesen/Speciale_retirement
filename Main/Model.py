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
            ('W',double),
            ('sigma_xi',double),            
            ('R',double),
            ('a_max',int32),
            ('a_phi',int32),
            ('Nxi',int32),
            ('Na',int32),

            # grids            
            ('grid_a',double[:]), # 1d array of doubles    
            ('xi',double[:]),        
            ('xi_w',double[:]),
            ('a_work',double[:,:]), # 2d array of doubles
            ('xi_work',double[:,:]),

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
            ('a',double[:,:]),
            ('xi',double[:,:]),
            ('psi',double[:,:])
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
        self.par.T = 20
        self.par.Tr = 10
        
        # preferences
        self.par.rho = 0.96
        self.par.beta = 0.98        
        self.par.alpha = 0.1
        self.par.gamma = 0.08

        # taste shocks
        self.par.sigma_eta = 0.435
        
        # income
        self.par.survival_probs = np.linspace(0.98, 0, self.par.T)
        self.par.Y = 1
        self.par.sigma_xi = 0.2
        
        # saving
        self.par.R = 1.03
        
        # grids
        self.par.a_max = 10
        self.par.a_phi = 1.1
        self.par.Nxi = 8
        self.par.Na = 150
        
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

        self.sol.c = np.nan*np.ones((self.par.T,self.par.Na,2))        
        self.sol.m = np.nan*np.zeros((self.par.T,self.par.Na,2))
        self.sol.v = np.nan*np.zeros((self.par.T,self.par.Na,2))
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
                
                post_decision.compute_work(t,self.sol,self.par)
                egm.solve_bellman_work(t,self.sol,self.par)
                print('Iteration', t)
                


#to debug code
from consav import runtools
runtools.write_numba_config(disable=1,threads=8)
model = RetirementModelClass(name='baseline',solmethod='egm')
model.solve()
print(model.sol.c)
