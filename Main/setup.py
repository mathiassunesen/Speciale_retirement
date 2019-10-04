# global modules
from numba import boolean, int32, int64, float64, double, njit, typeof
import numpy as np

# local modules
import transitions

def single_lists():

    parlist = [ # (name,numba type), parameters, grids etc.

            # boolean
            ('couple',boolean),

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
            ('m_com',double[:]),  
            ('grid_a',double[:]),     
            ('a_max',int32), 
            ('a_phi',double),
            ('Na',int32),
            ('poc',int32), 
            ('Nxi',int32), 

            # tax system
            ('fradrag',double),
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

            # simulation
            ('simT',int32), 
            ('simN',int32), 
            ('sim_seed',int32),  
            ('simM_init',double[:]),
            ('simMale',int32[:]),
            ('simStates',int32[:]), 

            # states
            ('ad_min',int32),
            ('ad_max',int32),            
            ('age_dif',int32[:]),
            ('male',int32[:]),  
            ('states',int32[:,:]), 
            ('inc_pre',double[:,:,:]),  
            ('labor',double[:,:,:,:]),            
            ('survival_arr',double[:,:]),  
            ('pension_arr',double[:,:,:,:,:]),

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
            ('xi_women_w',double[:]),

            # initial estimations (tuples)
            ('reg_labor_male',typeof((0.1, 0.2, 0.3, 0.4))),
            ('reg_labor_female',typeof((0.1, 0.2, 0.3, 0.4))),
            ('reg_survival_male',typeof((0.1, 0.2))),
            ('reg_survival_female',typeof((0.1, 0.2))),
            ('reg_pension_male',typeof((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))),
            ('reg_pension_female',typeof((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)))     
                                                 
        ]
        
    sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:,:,:]),    # 7d
            ('m',double[:,:,:,:,:,:,:]),
            ('v',double[:,:,:,:,:,:,:]),                     

            # interpolation
            ('c_plus_interp',double[:,:]),  # 2d
            ('v_plus_interp',double[:,:]),

            # post decision
            ('q',double[:,:]),              # 2d
            ('v_plus_raw',double[:,:])                            
        ]     

    simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:]),

            # states
            ('male',int32[:]),
            ('states',int32[:]),

            # dummies and probabilities
            ('alive',double[:,:]), 
            ('probs',double[:,:]), 
            ('ret_age',double[:]),

            # euler errors                      
            ('euler',double[:,:]),          

            # random shocks
            ('unif',double[:,:]),
            ('deadP',double[:,:]),
            ('inc_shock',double[:,:,:]),
            ('labor_array',double[:,:,:,:])

        ]

    return parlist,sollist,simlist


def couple_lists():

    single_par,single_sim = single_lists()[0,2]

    parlist = [ # (name,numba type), parameters, grids etc.

            # states     
            ('labor_c',double[:,:,:,:,:]),   

            # states for simulation
            ('simStatesH',int32[:]),
            ('simStatesW',int32[:]),

            # preference parameters
            ('pareto_w',double),
            ('v',double),            
            ('phi_0_male',double),
            ('phi_0_female',double),
            ('phi_1',double),

            # grids         
            ('Nxi_men',int32),
            ('Nxi_women',int32),                         
                
            # uncertainty/variance parameters
            ('sigma_eta',double), 
            ('sigma_xi_men',double), 
            ('sigma_xi_women',double),
            ('sigma_xi_cov',double),
            ('xi_men_corr',double[:]),
            ('xi_women_corr',double[:]),            
            ('w_corr',double[:])         
        ]

    parlist = parlist + single_par   

    sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:,:,:,:]),    # 8d
            ('m',double[:,:,:,:,:,:,:,:]),
            ('v',double[:,:,:,:,:,:,:,:]),                     

            # interpolation
            ('c_plus_interp',double[:,:]),  # 2d
            ('v_plus_interp',double[:,:]),

            # post decision
            ('q',double[:,:]),              # 2d
            ('v_plus_raw',double[:,:])                            
        ]              

    simlist = [ # (name,numba type), parameters, grids etc.

            # states for simulation
            ('states_h',int32[:]),
            ('states_w',int32[:])          
        ]

    simlist = simlist + single_sim

    return parlist,sollist,simlist


def init_sim(par,sim,dev=1.25,add=0.8):

        # initialize simulation
        np.random.seed(par.sim_seed)
        #par.simM_init = np.random.lognormal(-0.5*(dev**2),dev,size=par.simN)+add
        # hardcoded to simN=100000
        #par.simM_init = np.array([2.0]*100000)
        par.simM_init = np.array(np.array([1.6]*1095 +    # woman,erp=0,hs=0
                                          [2.2]*405 +     # woman,erp=0,hs=1
                                          [1.6]*35405 +   # woman,erp=1,hs=0
                                          [2.2]*13095 +   # woman,erp=1,hs=1
                                          [1.9]*2000 +    # man,erp=0,hs=0
                                          [2.6]*500 +     # man,erp=0,hs=1
                                          [1.9]*38000 +   # man,erp=1,hs=0
                                          [2.6]*9500))    # man,erp=1,hs=1

        # states
        if par.couple:
            par.simStatesH = np.random.randint(len(par.states),size=par.simN)
            par.simStatesW = np.random.randint(len(par.states),size=par.simN)
        else:
            # hardcoded to simN=100000
            #par.simMale = np.array([1]*100000)
            #par.simStates = np.array([3]*100000)
            par.simMale = np.array([0]*50000 + [1]*50000)   # equal split
            par.simStates = np.array([0]*1095 +     # woman,erp=0,hs=0
                                     [1]*405 +      # woman,erp=0,hs=1
                                     [2]*35405 +    # woman,erp=1,hs=0
                                     [3]*13095 +    # woman,erp=1,hs=1
                                     [0]*2000 +     # man,erp=0,hs=0
                                     [1]*500 +      # man,erp=0,hs=1
                                     [2]*38000 +    # man,erp=1,hs=0
                                     [3]*9500)      # man,erp=1,hs=1

        # random draws for simulation
        np.random.seed(par.sim_seed)        
        sim.unif = np.random.rand(par.simT,par.simN)                            
        sim.deadP = np.random.rand(par.simT,par.simN) 

        Tr = par.Tr
        sim.inc_shock = np.nan*np.zeros((Tr,len(par.male),par.simN))  
        sim.inc_shock[:,0,:] = np.random.lognormal(-0.5*(par.sigma_xi_women**2),par.sigma_xi_women,size=(Tr,par.simN))
        sim.inc_shock[:,1,:] = np.random.lognormal(-0.5*(par.sigma_xi_men**2),par.sigma_xi_men,size=(Tr,par.simN))


def RetirementSystem(model):

    # unpack and translate to model time
    par = model.par
    oap_age = transitions.inv_age(par.oap_age,par)
    two_year = transitions.inv_age(par.two_year,par)
    erp_age = transitions.inv_age(par.erp_age,par)
    assert(oap_age - erp_age == 5)
    assert(two_year - erp_age == 2)

    if model.year == 2008:

        if model.couple:
            model.ret_system = {
                oap_age-1:  [[0,0,0],[0,1,1],[0,2,2]],      # the list indicates how to solve the model in the given time period
                two_year+1: [[0,0,0],[1,1,1],[2,2,2]],      # 1. element is where to extract t+1 solution for husband
                two_year:   [[0,0,0],[1,1,1],[2,2,2]],      # 2. element is where to extract t+1 solution for wife
                two_year-1: [[1,1,0],[2,2,2]],              # 3. element is retirement status of husband
                erp_age:    [[0,1,0],[2,2,2]],              # 4. element is retirement status of wife
                erp_age-1:  [[2,2,0]]}                      # 5. element is where 

        else:
            # model.ret_system = {
            #     oap_age-1:  [[0], [1], [2]],
            #     two_year+1: [[0], [1], [2]],
            #     two_year:   [[0], [1], [2]],
            #     two_year-1: [     [1], [2]],
            #     erp_age:    [     [1], [2]],
            #     erp_age-1:  [          [2]]
            # } 
            model.ret_system = {  
                oap_age-1:  [[0, np.array([0,1])], [1, np.array([0])],      [2, np.array([0])]],
                two_year+1: [[0, np.array([0,1])], [1, np.array([0])],      [2, np.array([0])]],
                two_year:   [[0, np.array([0,1])], [1, np.array([0])],      [2, np.array([0])]],
                two_year-1: [                      [1, np.array([0,1])],    [2, np.array([0])]],
                erp_age:    [                      [1, np.array([0,1])],    [2, np.array([0])]],
                erp_age-1:  [                                               [2, np.array([0,1])]]
            }

    elif model.year == 2014:
        pass


def TaxSystem(model):

    if model.year == 2008:

        model.par.fradrag = 0.0#1.0 # denominated in 100.000 dkr
        model.par.tau_upper = 0.59
        model.par.tau_LMC = 0.08
        model.par.WD = 0.4
        model.par.WD_upper = 12300/100000
        model.par.tau_c = 0.2554
        model.par.y_low = 41000/100000
        model.par.y_low_m = 279800/100000
        model.par.y_low_u = 335800/100000
        model.par.tau_h = 0.08
        model.par.tau_l = 0.0548
        model.par.tau_m = 0.06
        model.par.tau_u = 0.15
        model.par.tau_max = model.par.tau_l + model.par.tau_m + model.par.tau_u + model.par.tau_c + model.par.tau_h - model.par.tau_upper         
    
    elif model.year == 2014:
        pass