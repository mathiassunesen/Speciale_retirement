# global modules
from numba import boolean, int32, int64, float64, double, njit, typeof
import numpy as np
import itertools

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

            # savings
            ('R',double),

            # grids          
            ('grid_a',double[:]),     
            ('a_max',int32), 
            ('a_phi',double),
            ('Na',int32),
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
            ('simStates',int32[:,:]),

            # states
            ('ad_min',int32),
            ('ad_max',int32),            
            ('AD',int32[:]),
            ('MA',int32[:]),  
            ('ST',int32[:,:]),  
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
            ('var_men',double), 
            ('var_women',double),
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
            ('avg_marg_u_plus',double[:,:,:,:,:,:,:]),  # 7d
            ('v_plus_raw',double[:,:,:,:,:,:,:])                      
        ]     

    simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:]),

            # states
            ('states',int32[:,:]),

            # dummies and probabilities
            ('alive',double[:,:,:]), 
            ('probs',double[:,:,:]), 
            ('RA',double[:,:]),

            # euler errors                      
            ('euler',double[:,:]),          

            # random shocks
            ('choiceP',double[:,:,:]),
            ('deadP',double[:,:,:]),
            ('inc_shock',double[:,:,:]),

        ]

    return parlist,sollist,simlist


def couple_lists():

    single_par,tmp,single_sim = single_lists()

    parlist = [ # (name,numba type), parameters, grids etc.

            # states     
            ('labor_c',double[:,:,:,:,:]),   

            # states for simulation
            ('simST_h',int32[:]),
            ('simST_w',int32[:]),

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
            ('cov',double),
            ('xi_men_corr',double[:]),
            ('xi_women_corr',double[:]),            
            ('w_corr',double[:])         
        ]

    parlist = parlist + single_par   

    sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:,:,:,:]),  # 8d
            ('m',double[:,:,:,:,:,:,:,:]),
            ('v',double[:,:,:,:,:,:,:,:]),                     

            # interpolation
            ('c_plus_interp',double[:,:]),  # 2d
            ('v_plus_interp',double[:,:]),

            # post decision
            ('q',double[:,:]),              # 2d
            ('v_raw',double[:,:])                                    
        ]              

    simlist = [ # (name,numba type), parameters, grids etc.

            # states for simulation
            ('ST_h',int32[:]),
            ('ST_w',int32[:])          
        ]

    simlist = simlist + single_sim

    return parlist,sollist,simlist


def init_sim(par,sim):

        # seed
        np.random.seed(par.sim_seed)
        
        # initialize simulation
        if par.couple:
            
            # initialize m 
            m_init = 10*np.ones(16)
            n_groups = (100000/16)*np.ones(16)
            par.simN = np.sum(n_groups) 
            par.simM_init = np.repeat(m_init,n_groups)

            # set states
            states = list(itertools.product([0,1,2,3], repeat=2))
            for i in range(len(states)):
                states[i] = (0,) + states[i]
            states = np.array(states)
            par.simStates = np.transpose(np.vstack((np.repeat(states[:,0],n_groups),
                                                    np.repeat(states[:,1],n_groups),
                                                    np.repeat(states[:,2],n_groups))))

            # random draws for simulation
            sim.choiceP = np.random.rand((par.simT,par.simN,2))
            sim.deadP = np.random.rand((par.simT,par.simN,2))  
            
            # random draws for labor income
            Tr = par.Tr
            sim.inc_shock = np.nan*np.zeros((Tr,par.simN,3))
            mu = -0.5*np.array([par.sigma_xi_women**2, par.sigma_xi_men**2])
            cov = np.array(([par.var_men, par.cov], [par.cov, par.var_women]))            
            sim.inc_shock[:,:,0] = np.exp(np.random.normal(mu[0],par.var_women,size=(Tr,par.simN)))
            sim.inc_shock[:,:,1] = np.exp(np.random.normal(mu[1],par.var_men,size=(Tr,par.simN)))
            sim.inc_shock[:,:,2] = np.exp(np.random.multivariate_normal(mu,cov,size=(Tr,par.simN)))        
        
        else:
        
            # initialize m
            m_init = np.array([10.0, 16.0, 10.0, 16.0, 11.0, 17.0, 11.0, 17.0])
            n_groups = np.array([1095, 405, 35405, 13095, 2000, 500, 38000, 9500])
            par.simN = np.sum(n_groups)
            par.simM_init = np.repeat(m_init,n_groups)

            # set states
            states = np.array([(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)])
            par.simStates = np.transpose(np.vstack((np.repeat(states[:,0],n_groups), 
                                                    np.repeat(states[:,1],n_groups))))

            # random draws for simulation       
            sim.choiceP = np.random.rand(par.simT,par.simN,1)                            
            sim.deadP = np.random.rand(par.simT,par.simN,1) 

            # random draws for labor income
            Tr = par.Tr
            sim.inc_shock = np.nan*np.zeros((Tr,par.simN,2))  
            sim.inc_shock[:,:,0] = np.exp(np.random.normal(-0.5*(par.var_women),np.sqrt(par.var_women),size=(Tr,par.simN)))
            sim.inc_shock[:,:,1] = np.exp(np.random.normal(-0.5*(par.var_men),np.sqrt(par.var_men),size=(Tr,par.simN)))            


def RetirementSystem(model):
    """ set up a dictionary (ret_system) that tells how to solve the model, when the solution has to be recalculated
        key: the relevant time period
        value: nested list, where each list tells what retirement status and choice set to solve for
    
    Args:
        model (class):

    Returns:
        stores ret_system in model
    """   

    # unpack and translate to model time
    par = model.par
    oap_age = transitions.inv_age(par.oap_age,par)
    two_year = transitions.inv_age(par.two_year,par)
    erp_age = transitions.inv_age(par.erp_age,par)
    assert(oap_age - erp_age == 5)
    assert(two_year - erp_age == 2)

    if model.year == 2008:

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