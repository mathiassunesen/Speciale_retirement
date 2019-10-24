# global modules
from numba import boolean, int32, int64, float64, double, njit, typeof
import numpy as np
import itertools
import pandas as pd

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
            ('T_oap',int32),
            ('T_erp',int32),
            ('T_two_year',int32),       

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
            ('inc_shock',double[:,:,:]),            

            # states
            ('ad_min',int32),
            ('ad_max',int32),            
            ('AD',int32[:]),
            ('MA',int32[:]),  
            ('ST',int32[:,:]),  
            ('iterator',int32[:,:]),
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
            ('alive',int32[:,:,:]), 
            ('probs',double[:,:,:]), 
            ('RA',int32[:,:]),

            # euler errors                      
            ('euler',double[:,:]),          

            # random shocks
            ('choiceP',double[:,:,:]),
            ('deadP',double[:,:,:]),
        ]

    return parlist,sollist,simlist


def couple_lists():

    single_par = single_lists()[0]

    parlist = [ # (name,numba type), parameters, grids etc.

            # states     
            ('labor_c',double[:,:,:,:,:]),   

            # simulation
            ('inc_shock_joint',double[:,:,:]),

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
            ('q',double[:,:,:,:,:,:,:,:]),              # 2d
            ('v_raw',double[:,:,:,:,:,:,:,:])               
            # ('q',double[:,:]),              # 2d
            # ('v_raw',double[:,:])                                    
        ]     

    simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:,:]),

            # states
            ('states',int32[:,:]),

            # dummies and probabilities
            ('alive',int32[:,:,:]), 
            ('probs',double[:,:,:]), 
            ('RA',int32[:,:]),

            # euler errors                      
            ('euler',double[:,:]),          

            # random shocks
            ('choiceP',double[:,:,:]),
            ('deadP',double[:,:,:]),
        ]                 

    return parlist,sollist,simlist

def TaxSystem(model):

    if model.year == 2008:

        model.par.fradrag = 0.0
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

def init_sim(model):

        # unpack 
        par = model.par
        sim = model.sim

        # seed
        np.random.seed(par.sim_seed)
        
        # initialize simulation
        if par.couple:
            
            # initialize m and states
            state_and_m(par,sim,perc_num=10)

            # random draws for simulation
            sim.choiceP = np.random.rand(par.simN,par.simT,2)
            sim.deadP = np.random.rand(par.simN,par.simT,2)  
            
            # random draws for labor income
            par.inc_shock = np.nan*np.zeros((par.simN,par.Tr,2))
            mu = -0.5*np.array([par.var_women, par.var_men])
            cov = np.array(([par.var_women, par.cov], [par.cov, par.var_men]))
            par.inc_shock[:,:,0] = np.exp(np.random.normal(mu[0],par.var_women,size=(par.simN,par.Tr)))
            par.inc_shock[:,:,1] = np.exp(np.random.normal(mu[1],par.var_men,size=(par.simN,par.Tr)))
            par.inc_shock_joint = np.exp(np.random.multivariate_normal(mu,cov,size=(par.simN,par.Tr)))        

            # precompute alive status
            sim.alive = np.ones((par.simN,par.simT,2),dtype=int)
            alive_w = sim.alive[:,:,0]
            alive_h = sim.alive[:,:,1]
            deadP_w = sim.deadP[:,:,0]
            deadP_h = sim.deadP[:,:,1]
            AD = sim.states[:,0]
            for t in range(par.simT):
                if t > 0:
                    alive_w[alive_w[:,t-1] == 0, t] = 0
                    alive_h[alive_h[:,t-1] == 0, t] = 0

                for ad in np.unique(AD):  
                    pi_h,pi_w = transitions.survival_look_up_c(t,ad,par) 
                    idx = np.nonzero(AD==ad)[0]                              
                    dead_w = idx[pi_w < deadP_w[idx,t]]
                    dead_h = idx[pi_h < deadP_h[idx,t]]
                    alive_w[dead_w,t] = 0
                    alive_h[dead_h,t] = 0
            
        else:
        
            # initialize m and states
            state_and_m(par,sim,perc_num=10)

            # # initialize m
            # m_init = np.array([10.0, 16.0, 10.0, 16.0, 11.0, 17.0, 11.0, 17.0])
            # n_groups = np.array([1095, 405, 35405, 13095, 2000, 500, 38000, 9500])
            # par.simN = np.sum(n_groups)
            # par.simM_init = np.repeat(m_init,n_groups)

            # # set states
            # states = np.array([(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)])
            # sim.states = np.transpose(np.vstack((np.repeat(states[:,0],n_groups), 
            #                                      np.repeat(states[:,1],n_groups))))

            # random draws for simulation       
            sim.choiceP = np.random.rand(par.simN,par.simT,1)                            
            sim.deadP = np.random.rand(par.simN,par.simT,1) 

            # random draws for labor income
            par.inc_shock = np.nan*np.zeros((par.simN,par.Tr,2))  
            par.inc_shock[:,:,0] = np.exp(np.random.normal(-0.5*(par.var_women),np.sqrt(par.var_women),size=(par.simN,par.Tr)))
            par.inc_shock[:,:,1] = np.exp(np.random.normal(-0.5*(par.var_men),np.sqrt(par.var_men),size=(par.simN,par.Tr)))            

            # precompute alive status
            sim.alive = np.ones((par.simN,par.simT,1),dtype=int)
            alive = sim.alive[:,:,0]
            deadP = sim.deadP[:,:,0]
            MA = sim.states[:,0]

            for t in range(par.simT):
                if t > 0:
                    alive[alive[:,t-1] == 0,t] = 0

                for ma in np.unique(MA):
                    pi = transitions.survival_look_up(t,ma,par)
                    idx = np.nonzero(MA==ma)[0]
                    dead = idx[pi < deadP[idx,t]]
                    alive[dead,t] = 0


def state_and_m(par,sim,perc_num=10):
    """ create states and initial wealth (m_init)"""

    if par.couple:

        # set states
        data = pd.read_excel('SASdata/couple_formue.xlsx')
        states = par.iterator
        n_groups = (data['Frac'].to_numpy()*par.simN).astype(int)
        n_groups[-1] = par.simN-np.sum(n_groups[:-1])   # assure it sums to simN
        sim.states = np.transpose(np.vstack((np.repeat(states[:,0],n_groups),
                                             np.repeat(states[:,1],n_groups),
                                             np.repeat(states[:,2],n_groups))))
    
    else:
        
        # set states
        data = pd.read_excel('SASdata/single_formue.xlsx')
        states = par.iterator[:,1:]
        n_groups = (data['Frac'].to_numpy()*par.simN).astype(int)
        n_groups[-1] = par.simN-np.sum(n_groups[:-1])   # to assure it sums to simN
        sim.states = np.transpose(np.vstack((np.repeat(states[:,0],n_groups),
                                             np.repeat(states[:,1],n_groups))))
        
    # set m_init
    m_init = np.zeros(len(sim.states))
    idx = np.concatenate((np.zeros(1), np.cumsum(n_groups))).astype(int)
    percentiles = np.linspace(0,100,perc_num+1).astype(int)
    bins = data[list(percentiles)].to_numpy()
    for i in range(n_groups.size):
        m_init[idx[i]:idx[i+1]] = pc_sample(n_groups[i], percentiles, bins[i])
    par.simM_init = m_init
        
def pc_sample(N,percentiles,bins):
    """ N samples from a dsitribution given its percentiles and bins (assumes equal spacing between percentiles)"""
    diff = np.diff(percentiles)
    assert np.allclose(diff[0],diff)
    n = int(N/diff.size)
    draws = np.random.uniform(low=bins[:-1], high=bins[1:], size=(n,diff.size)).ravel()
    return np.concatenate((draws, np.random.uniform(low=bins[0], high=bins[-1], size=(N-n*diff.size)))) # to assure we return N samples                        