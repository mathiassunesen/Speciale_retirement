# global modules
from numba import boolean, int32, int64, float64, double, njit, typeof
import numpy as np
import itertools
import pandas as pd

# consav package
from consav import misc 

# local modules
import transitions
import funs

def single_lists():

    parlist = [ # (name,numba type), parameters, grids etc.

            # boolean
            ('couple',boolean),

            # misc
            ('denom',double),
            ('tol',double),

            # time parameters
            ('start_T',int32),
            ('end_T',int32),
            ('forced_T',int32),
            ('simT',int32),

            # simulation            
            ('sim_seed',int32),
            ('simN',int32),

            # savings
            ('R',double),

            # grids           
            ('a_max',int32), 
            ('a_phi',double),
            ('Na',int32),
            ('Nxi',int32), 

            # states  
            ('MA',int32[:]),  
            ('ST',int32[:,:]),  
            ('AD',int32[:]),

            # preference parameters
            ('rho',double), 
            ('beta',double),
            ('alpha_0_male',double), 
            ('alpha_0_female',double),
            ('alpha_1',double),
            ('gamma',double),            

            # uncertainty/variance parameters
            ('sigma_eta',double), 
            ('var',double[:]), 

            # initial estimations
            ('reg_labor_male',double[:]),
            ('reg_labor_female',double[:]),
            ('reg_survival_male',double[:]),
            ('reg_survival_female',double[:]),
            ('pension_male',double[:]),
            ('pension_female',double[:]),   

            # tax system
            ('IRA_tax',double),
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
            ('oap_B',double),
            ('A_i',double[:]),
            ('tau_i',double[:]),
            ('D_s',double),
            ('D_i',double[:]),
            ('ERP_low',double),
            ('ERP_high',double),
            ('ERP_2',double),

            # model time
            ('T',int32),
            ('Tr',int32),   
            ('T_oap',int32),
            ('T_erp',int32),
            ('T_two_year',int32),       
            ('ad_min',int32),
            ('ad_max',int32),
            ('iterator',int32[:,:]),

            # grids
            ('grid_a',double[:]),
            ('xi',double[:,:]),
            ('xi_w',double[:,:]),

            # precompute
            ('survival',double[:,:,:]),
            ('oap',double[:]),
            ('labor',double[:,:,:,:]),
            ('erp',double[:,:,:,:]),

            # simulation
            ('simM_init',double[:])            

        ]
        
    sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:,:,:]),
            ('m',double[:]),
            ('v',double[:,:,:,:,:,:,:]),      

            # post decision
            ('avg_marg_u_plus',double[:,:,:,:,:,:,:]), 
            ('v_plus_raw',double[:,:,:,:,:,:,:])                      
        ]     

    simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:]),

            # dummies, probabilities and euler errors
            ('probs',double[:,:,:]), 
            ('RA',int32[:,:]),
            ('euler',double[:,:]),   

            # setup
            ('choiceP',double[:,:,:]),
            ('deadP',double[:,:,:]),
            ('shocks',double[:,:,:]),
            ('alive',int32[:,:,:]),      
            ('states',int32[:,:])

        ]

    return parlist,sollist,simlist


def couple_lists():

    single_par = single_lists()[0]

    parlist = [ # (name,numba type), parameters, grids etc.

            # states     
            ('labor_ind',double[:,:,:,:,:,:,:,:]),
            ('labor_joint',double[:,:,:,:,:,:]),
            ('pension_ind',double[:,:,:,:,:,:,:]),
            ('pension_joint',double[:,:,:,:,:,:]),
               
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
            ('v_plus_interp',double[:,:])                        
        ]     

    simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:,:]),

            # states
            ('states',int32[:,:]),

            # moments
            ('moments',double[:,:,:]),

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
    """ Tax system for either 2008 or 2014 """

    # unpack
    par = model.par

    if model.year == 2008:

        par.IRA_tax = 0.4               # tax for IRA (kapitalpension)
        par.fradrag = 0.0               # deduction for old workers (policy proposal)
        par.tau_upper = 0.59            # maximum tax rate (skatteloft)
        par.tau_LMC = 0.08              # labor market contribution (arbejdsmarkedsbidrag)
        par.WD = 0.4                    # working deduction (beskæftigelsesfradrag)
        par.WD_upper = 12300/par.denom  # maximum deduction possible (beskæftigelsesfradrag, maksimal)
        par.tau_c = 0.2554              # average county-specific rate (including 0.073 in church tax)
        par.y_low = 41000/par.denom     # amount deductible from all income (personfradrag)
        par.y_low_m = 279800/par.denom  # amount deductible from middle tax bracket (mellemskattegrænse)
        par.y_low_u = 335800/par.denom  # amount deductible from top tax bracket (topskattegrænse)
        par.tau_h = 0.08                # health contribution tax (sundhedsbidrag)
        par.tau_l = 0.0548              # tax rate in lowest tax bracket (bundskat)
        par.tau_m = 0.06                # tax rate in middle tax bracket (mellemskat)
        par.tau_u = 0.15                # tax rate in upper tax bracket (topskat)
        par.tau_max = par.tau_l + par.tau_m + par.tau_u + par.tau_c + par.tau_h - par.tau_upper         
    
    elif model.year == 2014:
        pass  

def RetirementSystem(model): 
    """ Retirement system for either 2008 or 2014 """     

    # unpack
    par = model.par

    if model.year == 2008:

        par.oap_age = 65
        par.two_year = 62
        par.erp_age = 60
        par.oap_B = 61152/par.denom                             # base rate
        par.A_i = np.array([61560, 28752, 28752])/par.denom     # Maximum OAP_A
        par.tau_i = np.array([0.3, 0.3, 0.15])                  # marginal reduction in OAP_A
        par.D_s = 179400/par.denom                              # maximum deduction in spousal income
        par.D_i = np.array([57300, 115000, 115000])/par.denom   # maximum deduction regarding OAP_A
        par.ERP_low = 12600/par.denom                           # deduction
        par.ERP_high = 166400/par.denom                         # maximum erp if two year rule is not satisfied
        par.ERP_2 = 182780/par.denom                            # erp with two year rule

    elif model.year == 2014:
        pass

def model_time(par):
    """ translate variables to model time and generate iterator for solving"""

    par.T = par.end_T - par.start_T + 1                  # total time periods
    par.Tr = par.forced_T - par.start_T + 1              # total time periods to forced retirement
    par.T_oap = transitions.inv_age(par.oap_age,par)
    par.T_erp = transitions.inv_age(par.erp_age,par)
    par.T_two_year = transitions.inv_age(par.two_year,par)
    par.ad_min = abs(min(par.AD))
    par.ad_max = max(par.AD)
    if par.couple:
        par.iterator = create_iterator([par.AD,par.ST,par.ST],3) 
    else:
        par.iterator = create_iterator([par.AD,par.MA,par.ST],3)       

def create_iterator(lst,num):
    indices = 0
                 
    if num == 3:
        iterator = np.zeros((len(lst[0])*len(lst[1])*len(lst[2]),num),dtype=int)        
        for x in lst[0]:
            for y in range(len(lst[1])):
                for z in range(len(lst[2])):
                    iterator[indices] = (x,y,z)
                    indices += 1
    
    return iterator                 

def grids(par):
    """ construct grids for states and shocks """

    # a. a-grid (unequally spaced vector of length Na)
    par.grid_a = misc.nonlinspace(par.tol,par.a_max,par.Na,par.a_phi)
        
    # b. shocks (quadrature nodes and weights for GaussHermite)
    par.xi = np.nan*np.zeros((len(par.MA),par.Nxi))
    par.xi_w = np.nan*np.zeros((len(par.MA),par.Nxi))
    for ma in range(len(par.MA)):
        par.xi[ma],par.xi_w[ma] = funs.GaussHermite_lognorm(par.var[ma],par.Nxi)
        
    # # c. correlated shocks for joint labor income (only for couples)
    # if par.couple:                      
    #     par.xi_men_corr,par.xi_women_corr,par.w_corr = funs.GH_lognorm_corr(par.var_men,par.var_women,par.cov,par.Nxi_men,par.Nxi_women)    

def init_sim(model):
    """ initialize simulation (wrapper) """

    # set seed
    np.random.seed(model.par.sim_seed) 

    if model.couple:
        pass

    else:
        init_sim_single(model)

def init_sim_single(model):
    """ initialize simulation for singles """

    # unpack 
    par = model.par
    sim = model.sim

    # initialize m and states
    state_and_m(par,sim,perc_num=10)

    # random draws       
    sim.choiceP = np.random.rand(par.simN,par.simT,1)                            
    sim.deadP = np.random.rand(par.simN,par.simT,1) 
    sim.shocks = np.nan*np.zeros((par.simN,par.Tr,2))
    for ma in range(len(par.MA)):
        sim.shocks[:,:,ma] = np.exp(np.random.normal(-0.5*par.var[ma], np.sqrt(par.var[ma]), size=(par.simN,par.Tr)))

    # precompute
    MA = sim.states[:,0]
    MAx = np.unique(MA)
    ST = sim.states[:,1]
    STx = np.unique(ST)
             
    # alive status
    sim.alive = np.ones((par.simN,par.simT,1),dtype=int)
    alive = sim.alive[:,:,0]
    deadP = sim.deadP[:,:,0]
    for t in range(par.simT):
        if t > 0:
            alive[alive[:,t-1] == 0,t] = 0

        for ma in MAx:
            for st in STx:
                pi = transitions.survival_lookup_single(t,ma,st,par)
                idx = np.nonzero((MA==ma) & (ST==st))[0]
                dead = idx[pi < deadP[idx,t]]
                alive[dead,t] = 0

def init_sim_couple(model):

    # unpack 
    par = model.par
    sim = model.sim
            
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

def state_and_m(par,sim,perc_num=10):
    """ create states and initial wealth (m_init) by loading in relevant information from SASdata"""

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