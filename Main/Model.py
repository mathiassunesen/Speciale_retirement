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
from numba.typed import Dict
import itertools

# consav package
from consav import linear_interp 
from consav import misc 
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
    
    def __init__(self,name='baseline',couple=False,year=2008,load=False,single_kwargs={},**kwargs):

        # a. store args
        self.name = name 
        self.couple = couple
        self.year = year

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
            self.Single = RetirementClass(name=name+'_single',year=year,**single_kwargs)
    
    def pars(self,**kwargs):
        """ define baseline values and update with user choices

        Args:

             **kwargs: change to baseline parameters in .par

        """   
        # boolean
        if self.couple:
            self.par.couple = True
        else:
            self.par.couple = False

        # time parameters
        self.par.start_T = 57   # start age
        self.par.end_T = 110    # end age
        self.par.forced_T = 77  # forced retirement age

        # savings
        self.par.R = 1.03       # interest rate             

        # grids
        self.par.a_max = 100    # 10 mio. kr. denominated in 100.000 kr
        self.par.a_phi = 1.5    # curvature of grid
        self.par.Na = 200       # total no. of points in a_grid=Na+poc
        self.par.Nxi = 8        # no. of GH-points
        if self.couple:
            self.par.Nxi_men = 4
            self.par.Nxi_women = 4 
        
        # tax system
        setup.TaxSystem(self)

        # retirement
        self.par.oap_age = 65
        self.par.two_year = 62
        self.par.erp_age = 60
        self.par.oap_base = 61152   # base rate
        self.par.oap_add = 61560    # tillæg
        self.par.erp_high = 182780  # erp with two year rule

        # misc
        self.par.tol = 1e-6

        # simulation
        self.par.simN = 100000
        self.par.sim_seed = 1998

        # states
        self.par.male = np.arange(2)                                            # same as [0,1]            
        self.par.states = np.array(list(itertools.product([0, 1], repeat=2)))   # 2 dummy states = 4 combinations                   
        if self.couple:
            #self.par.age_dif = np.arange(-4,5)                                  # same as [-4,-3,-2-,1,0,1,2,3,4]
            self.par.age_dif = np.arange(-1,2)
        else:
            self.par.age_dif = np.arange(1)                                     # same as [0]
        self.par.ad_min = abs(min(self.par.age_dif))
        self.par.ad_max = max(self.par.age_dif)

        # preference parameters
        self.par.rho = 0.96                         # crra
        self.par.beta = 0.98                        # time preference
        self.par.alpha_0_male = 0.160               # constant, own leisure
        self.par.alpha_0_female = 0.119             # constant, own leisure
        self.par.alpha_1 = 0.053                    # high skilled, own leisure
        self.par.gamma = 0.08                       # bequest motive
        if self.couple:
            self.par.pareto_w = 0.5                 # pareto weight 
            self.par.v = 0.048                      # equivalence scale                    
            self.par.phi_0_male = 1.187             # constant, joint leisure
            self.par.phi_0_female = 1.671           # constant, joint leisure
            self.par.phi_1 = -0.621                 # high skilled, joint leisure                      

        # uncertainty/variance parameters
        self.par.sigma_eta = 0.435                  # taste shock
        self.par.sigma_xi_men = np.sqrt(0.544)      # income shock
        self.par.sigma_xi_women = np.sqrt(0.399)    # income shock
        if self.couple:
            self.par.sigma_xi_cov = 0.011           # covariance of income shocks            

        # initial estimations
        if self.couple:
            self.par.reg_labor_male =(-5.999, 0.262, 0.629, -0.532)
            self.par.reg_labor_female = (-4.002, 0.318, 0.544, -0.453)            
            self.par.reg_survival_male = (-10.338, 0.097)
            self.par.reg_survival_female = (-11.142, 0.103)
            self.par.reg_pension_male = (-41.161, 0.072, -0.068, 0.069, 
                                           8.864, -0.655, 0.016)
            self.par.reg_pension_female = (-19.000, 0.039, -0.037, 0.131,
                                             4.290, -0.327, 0.008)             
        else:
            self.par.reg_labor_male = (-15.956, 0.230, 0.934, -0.770)       # order is: cons, high_skilled, age, age2
            self.par.reg_labor_female = (-18.937, 0.248, 1.036, -0.856)     # order is: cons, high_skilled, age, age2 
            self.par.reg_survival_male = (-10.338, 0.097)                   # order is: cons, age
            self.par.reg_survival_female = (-11.142, 0.103)                 # order is: cons, age
            self.par.reg_pension_male = (-57.670, 0.216, -0.187, 0.142,     # order is: cons, age, age2, high_skilled
                                          12.057, -0.920, 0.023)            #           log_wealth, log_wealth2, log_wealth3
            self.par.reg_pension_female = (-47.565, 0.098, -0.091, 0.185,   # order is: cons, age, age2, high_skilled
                                            10.062, -0.732, 0.018)          #           log_wealth, log_wealth2, log_wealth3


        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val

        
        # c. update parameters, which depends on other parameters
        self.par.T = self.par.end_T - self.par.start_T + 1                  # total time periods
        self.par.Tr = self.par.forced_T - self.par.start_T + 1              # total time periods to forced retirement
        self.par.simT = self.par.T                                          # total time periods in simulation
        setup.RetirementSystem(self)        
        
        # d. initialize simulation
        setup.init_sim(self.par,self.sim)

        # e. setup_grids
        self.setup_grids()

        # f. precompute state variables from initial estimations
        transitions.labor_precompute(self.par)
        transitions.survival_precompute(self.par) 
        transitions.pension_precompute(self.par)


    def setup_grids(self):
        """ construct grids for states and shocks """

        # a. post-decision states (unequally spaced vector of length Na)
        self.par.grid_a = misc.nonlinspace(self.par.tol,self.par.a_max,self.par.Na,self.par.a_phi)
        
        # b. shocks (quadrature nodes and weights for GaussHermite)
        self.par.xi_men,self.par.xi_men_w = funs.GaussHermite_lognorm(self.par.sigma_xi_men,self.par.Nxi)
        self.par.xi_women,self.par.xi_women_w = funs.GaussHermite_lognorm(self.par.sigma_xi_women,self.par.Nxi)   
        
        # c. correlated shocks for joint labor income (only for couples)
        if self.couple:
            self.par.xi_men_corr,self.par.xi_women_corr,self.par.w_corr = funs.GH_lognorm_corr(self.par.sigma_xi_men,self.par.sigma_xi_women,self.par.sigma_xi_cov,self.par.Nxi_men,self.par.Nxi_women)    

    #########
    # solve #
    #########
    def _solve_prep(self,recompute):
        """ allocate memory for solution """ 

        if recompute:
            self.setup_grids()
            transitions.labor_precompute(self.par)
            transitions.survival_precompute(self.par) 
            transitions.pension_precompute(self.par)            

        # prep
        num_ad = len(self.par.age_dif)          # number of age differences             
        num_st = len(self.par.states)           # number of states
        num_grid = self.par.Na                  # number of points in grid           
        num_erp = 3                             # number of erp status

        if self.couple:
            num_d = 4                           # number of choices
        else:
            num_gen = len(self.par.male)        # number of gender
            num_d = 2                           # number of choices
            
        # solution
        if self.couple:
            self.sol.c = np.nan*np.zeros((self.par.T,num_ad,num_st,num_st,num_erp,num_erp,num_d,num_grid))   
            self.sol.m = np.nan*np.zeros((self.par.T,num_ad,num_st,num_st,num_erp,num_erp,num_d,num_grid))
            self.sol.v = np.nan*np.zeros((self.par.T,num_ad,num_st,num_st,num_erp,num_erp,num_d,num_grid))   
        else:
            self.sol.c = np.nan*np.zeros((self.par.T,num_ad,num_gen,num_st,num_erp,num_d,num_grid))   
            self.sol.m = np.nan*np.zeros((self.par.T,num_ad,num_gen,num_st,num_erp,num_d,num_grid))
            self.sol.v = np.nan*np.zeros((self.par.T,num_ad,num_gen,num_st,num_erp,num_d,num_grid))            

        # interpolation
        self.sol.c_plus_interp = np.nan*np.zeros((num_d,num_grid))
        self.sol.v_plus_interp = np.nan*np.zeros((num_d,num_grid)) 

        # post decision
        self.sol.q = np.nan*np.zeros((num_d,num_grid))
        self.sol.v_plus_raw = np.nan*np.zeros((num_d,num_grid))

    def solve(self,recompute=False):
        """ solve the model """

        # a. allocate solution
        if self.couple: # if couple first prep the single model
            self.Single._solve_prep(recompute)  
        self._solve_prep(recompute)

        # b. solve the model
        if self.couple: # if couple solve both models
            solution.solve(self.Single)
            solution.solve_c(self)
        else:           # else only solve single model
            solution.solve(self)
                            

    ############
    # simulate #
    ############
    def _simulate_prep(self,recompute):
        """ allocate memory for simulation and draw random numbers """

        # recompute if var changes have been made
        if recompute:
            setup.init_sim(self.par,self.sim)

        # solution
        self.sim.c = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.m = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.a = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.d = np.nan*np.zeros((self.par.simT,self.par.simN))

        # dummies and probabilities
        self.sim.alive = np.ones((self.par.simT,self.par.simN))             # dummy for alive
        self.sim.probs = np.zeros((self.par.simT,self.par.simN))            # retirement probs
        self.sim.ret_age = 2*np.ones(self.par.simN)                         # retirement status

        # initialize m and d
        self.sim.m[0,:] = self.par.simM_init                                # has computed in setup    
        self.sim.d[0,:] = np.ones(self.par.simN)                            # all is working at t=0
        
        # states
        self.sim.male = self.par.simMale                                    # has been computed in setup
        self.sim.states = self.par.simStates                                # has been computed in setup

        # euler errors
        self.sim.euler = np.nan*np.zeros((self.par.simT-1,self.par.simN))


    def simulate(self,euler=False,recompute=False):
        """ simulate model """

        # a. allocate memory and draw random numbers 
        self._simulate_prep(recompute)
        
        # b. simulate
        simulate.lifecycle(self.sim,self.sol,self.par)

        # c. euler errors
        if euler:
            simulate.euler_error(self.sim,self.sol,self.par)


# #to debug code
# Na = 20
# single_kwargs = {'Na':Na}
# data = RetirementClass(couple=True,single_kwargs=single_kwargs,Na=Na)
# data.solve()
# #data.simulate(euler=True)


# data = RetirementClass()
# data.solve()
# data.simulate()