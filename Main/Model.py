# -*- coding: utf-8 -*-
"""RetirementModel"""

##############
# 1. imports #
##############

# global modules
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import time
import numpy as np
from numba import boolean, int32, int64, float64, double, njit, prange, typeof
import itertools

# consav package
from consav import linear_interp 
from consav import ModelClass 

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
import setup

############
# 2. model #
############

class RetirementClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def __init__(self,name='baseline',couple=False,year=2008,Thomas=False,couple_finance=True,
                 load=False,single_kwargs={},**kwargs):

        # a. store args
        self.name = name 
        self.couple = couple
        self.year = year
        self.Thomas = Thomas
        self.couple_finance = couple_finance

        # b. subclasses 
        if couple:
            parlist,sollist,simlist = setup.couple_lists()
        else:
            parlist,sollist,simlist = setup.single_lists()
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        # c. load
        if load:
            self.load()
        else:
            self.pars(**kwargs)

        # d. if couple also create a single class
        if couple:
            single_kwargs['start_T'] = self.par.start_T - self.par.ad_min # add some extra time periods in the bottom        
            single_kwargs['var'] = self.par.var
            single_kwargs['reg_labor_male'] = self.par.reg_labor_male
            single_kwargs['reg_labor_female'] = self.par.reg_labor_female
            single_kwargs['g_adjust'] = self.par.g_adjust
            single_kwargs['priv_pension_female'] = self.par.priv_pension_female
            single_kwargs['priv_pension_male'] = self.par.priv_pension_male            
            single_kwargs['simN'] = 100   # we don't need sim so we don't want to waste memory
            single_kwargs['simT'] = 1
            self.Single = RetirementClass(name=name+'_single',year=year,**single_kwargs)
    
    def pars(self,**kwargs):
        """ define baseline values and update with user choices

        Args:

             **kwargs: change to baseline parameters in .par

        """   
        # boolean
        self.par.couple = self.couple
        self.par.Thomas = self.Thomas
        self.par.couple_finance = self.couple_finance

        # misc
        self.par.denom = 1e5       # all monetary variables are denominated in 100.000 DKR
        self.par.tol = 1e-6

        # time parameters
        self.par.start_T = 57                                   # start age
        self.par.end_T = 110                                    # end age
        self.par.forced_T = 77                                  # forced retirement age
        self.par.simT = self.par.end_T - self.par.start_T + 1   # total time periods in simulation     

        # simulation
        self.par.sim_seed = 2019
        self.par.simN = int(1e5)            

        # savings
        self.par.R = 1.03       # interest rate             

        # grids
        self.par.a_max = 150    # 15 mio. kr. denominated in 100.000 kr
        self.par.a_phi = 1.1    # curvature of grid
        self.par.Na = 200       # no. of points in a grid
        self.par.Nxi = 5        # no. of GH-points
        if self.couple:
            self.par.Nxi_men = 5
            self.par.Nxi_women = 5 

        # states
        if self.couple:
            self.par.AD = np.array([-4,-3,-2,-1,0,1,2,3,4])
        self.par.MA = np.array([0,1])            
        self.par.ST = np.array(list(itertools.product([0, 1], repeat=2)))   # 2 dummy states = 4 combinations                   

        # preference parameters
        self.par.rho = 0.96                         # crra
        self.par.beta = 0.98                        # time preference
        self.par.alpha_0_male = 0.4                 # constant, own leisure
        self.par.alpha_0_female = 0.2               # constant, own leisure
        self.par.alpha_1 = 0.0                      # high skilled, own leisure
        self.par.gamma = 0.08                       # bequest motive
        self.par.v = 0.048                          # equivalence scale          
        if self.couple:
            self.par.pareto_w = 0.5                 # pareto weight 
            if self.Thomas:
                self.par.phi_0_male = 1.187         # constant, joint leisure
                self.par.phi_0_female = 1.671       # constant, joint leisure
                self.par.phi_1 = -0.621             # high skilled, joint leisure   
            else:                  
                self.par.phi_0_male = 1             # constant, joint leisure
                self.par.phi_0_female = 1           # constant, joint leisure
                self.par.phi_1 = 0.0                # high skilled, joint leisure                      

        # uncertainty/variance parameters
        self.par.sigma_eta = 0.435                  # taste shock
        if self.couple:
            if self.Thomas:
                self.par.var = np.array([0.347, 0.288]) # income shocks (women first)
                self.par.cov = 0.011                    # covariance of income shocks
            else:
                self.par.var = np.array([0.202, 0.161]) # income shocks (women first)
                self.par.cov = 0.002                     # covariance of income shocks    
        else:
            if self.Thomas:
                self.par.var = np.array([0.399, 0.544]) # income shocks (women first)
            else:
                self.par.var = np.array([0.241, 0.123]) # income shocks (women first)

        # initial estimations
        self.par.pi_adjust_f =              0.216/100  
        self.par.pi_adjust_m =              0.422/100        
        self.par.reg_survival_male =        np.array((-10.338, 0.097))          # order is: cons, age
        self.par.reg_survival_female =      np.array((-11.142, 0.103))          # order is: cons, age

        if self.couple:

            # labor market income
            if self.Thomas:
                self.par.reg_labor_male =       np.array((-5.999, 0.262, 0.629, -0.532))        # order is: cons, high_skilled, age, age2
                self.par.reg_labor_female =     np.array((-4.002, 0.318, 0.544, -0.453))        # order is: cons, high_skilled, age, age2    
            else:
                self.par.reg_labor_male =       np.array((1.166, 0.360, 0.432, -0.406))     # order is: cons, high_skilled, age, age2   
                self.par.reg_labor_female =     np.array((4.261, 0.326, 0.303, -0.289))   # order is: cons, high_skilled, age, age2

            # private pension
            self.par.g_adjust = 0.75
            self.par.priv_pension_female =  728*1000/self.par.denom
            self.par.priv_pension_male =    1236*1000/self.par.denom            

        else:

            # labor market income
            if self.Thomas:
                self.par.reg_labor_male =       np.array((-15.956, 0.230, 0.934, -0.770))       # order is: cons, high_skilled, age, age2
                self.par.reg_labor_female =     np.array((-18.937, 0.248, 1.036, -0.856))       # order is: cons, high_skilled, age, age2 
            else:
                self.par.reg_labor_male =       np.array((3.374, 0.318, 0.310, -0.261))       # order is: cons, high_skilled, age, age2
                self.par.reg_labor_female =     np.array((1.728, 0.299, 0.342, -0.278))       # order is: cons, high_skilled, age, age2                 
            
            # private pension
            self.par.g_adjust = 0.75
            self.par.priv_pension_female =  744*1000/self.par.denom
            self.par.priv_pension_male =    682*1000/self.par.denom            


        # tax and retirement system
        setup.TaxSystem(self)
        setup.RetirementSystem(self)            

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val

        # c. precompute
        self.recompute()

    def recompute(self):
        """ recompute precomputations if institutional variables have been changed """ 
        # 1. translate to model time and setup grids        
        setup.model_time(self.par)
        setup.grids(self.par)
        
        # 2. precompute and initialize simulation (sensitive to the order)
        transitions.precompute_survival(self.par)
        setup.init_sim(self.par,self.sim)
        if self.couple:
            transitions.precompute_inc_couple(self.par)
        else:
            transitions.precompute_inc_single(self.par)

    #########
    # solve #
    #########
    def _solve_prep(self,recompute):
        """ allocate memory for solution """ 

        if recompute:
            self.recompute()

        # prep
        T = self.par.T          
        NST = len(self.par.ST)          # number of states
        Na = self.par.Na                # number of points in grid           
        NRA = 3                         # number of retirement status

        if self.couple:
            NAD = len(self.par.AD)      # number of age differences               
            ND = 4                      # number of choices

            # solution
            self.sol.c = np.nan*np.zeros((T,NAD,NST,NST,NRA,NRA,ND,Na))   
            self.sol.m = self.par.grid_a    # common grid
            self.sol.v = np.nan*np.zeros((T,NAD,NST,NST,NRA,NRA,ND,Na))        

        else:
            NMA = len(self.par.MA)      # number of gender
            ND = 2                      # number of choices

            # solution
            self.sol.c = np.nan*np.zeros((T,NMA,NST,NRA,ND,Na))   
            self.sol.m = self.par.grid_a    # common grid
            self.sol.v = np.nan*np.zeros((T,NMA,NST,NRA,ND,Na))     

            # post decision
            self.sol.avg_marg_u_plus = np.nan*np.zeros((T,NMA,NST,NRA,ND,Na))
            self.sol.v_plus_raw = np.nan*np.zeros((T,NMA,NST,NRA,ND,Na)) 

    def solve(self,recompute=False):
        """ solve the model """

        if self.couple:

            # allocate solution
            self.Single._solve_prep(recompute)
            self._solve_prep(recompute)
        
            # solve model
            solution.solve(self.Single.sol,self.Single.par)
            solution.solve_c(self.sol,self.Single.sol,self.par)

        else:

            # allocate solution
            self._solve_prep(recompute)

            # solve model
            solution.solve(self.sol,self.par)
        
    ############
    # simulate #
    ############
    def _simulate_prep(self,accuracy,tax):
        """ allocate memory for simulation """

        if self.couple:

            extend = self.par.ad_min + self.par.ad_max

            # solution
            self.sim.c = np.nan*np.zeros((self.par.simN,self.par.simT))
            self.sim.a = np.nan*np.zeros((self.par.simN,self.par.simT))
            self.sim.d = np.nan*np.zeros((self.par.simN,self.par.simT+extend,2))

            # misc
            self.sim.probs = np.nan*np.zeros((self.par.simN,self.par.simT+extend,2))  
            self.sim.RA = 2*np.ones((self.par.simN,2),dtype=int)
            self.sim.euler = np.nan*np.zeros((self.par.simN,self.par.simT-1))
            self.sim.GovS = np.nan*np.zeros((self.par.simN,self.par.simT))            

            # booleans
            self.sim.accuracy = accuracy
            self.sim.tax = tax

        else:

            # solution
            self.sim.c = np.nan*np.zeros((self.par.simN,self.par.simT))
            self.sim.a = np.nan*np.zeros((self.par.simN,self.par.simT))
            self.sim.d = np.nan*np.zeros((self.par.simN,self.par.simT))

            # misc
            self.sim.probs = np.nan*np.zeros((self.par.simN,self.par.simT))  
            self.sim.RA = 2*np.ones((self.par.simN),dtype=int)
            self.sim.euler = np.nan*np.zeros((self.par.simN,self.par.simT-1))
            self.sim.GovS = np.nan*np.zeros((self.par.simN,self.par.simT))

            # booleans
            self.sim.accuracy = accuracy
            self.sim.tax = tax

            # initialize d
            self.sim.d[:,0] = 1

    def simulate(self,accuracy=False,tax=False):
        """ simulate model """

        if self.couple:

            # allocate memory
            self._simulate_prep(accuracy,tax)

            # simulate model
            simulate.lifecycle_c(self.sim,self.sol,self.Single.sol,self.par,self.Single.par)

        else:

            # allocate memory
            self._simulate_prep(accuracy,tax)

            # simulate model
            simulate.lifecycle(self.sim,self.sol,self.par)
    

# Single = RetirementClass()
# Single.solve()
# Single.simulate()


# test = RetirementClass()
# test._simulate_prep(False,False)
# test.solve()

# single_kwargs = {'Na':20}
# data = RetirementClass(couple=True, single_kwargs=single_kwargs, Na=20)
# data.solve()

# test = RetirementClass(couple=True)
# test.solve()

# test = RetirementClass(couple=True, load=True)
# test.Single = RetirementClass(name='baseline_single', load=True)
# # test.par.simT=12
# test.recompute()
# test.simulate(tax=True)
