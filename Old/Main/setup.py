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
            ('Thomas',boolean),
            ('pension_no_tax',boolean),

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
            ('v',double),         

            # uncertainty/variance parameters
            ('sigma_eta',double), 
            ('var',double[:]), 

            # initial estimations
            ('pi_adjust_f',double),
            ('pi_adjust_m',double),
            ('reg_survival_male',double[:]),
            ('reg_survival_female',double[:]),
            ('reg_labor_male',double[:]),
            ('reg_labor_female',double[:]),
            ('g_adjust',double),
            ('priv_pension_female',double),
            ('priv_pension_male',double),            

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
            ('B',double),
            ('y_B',double),
            ('tau_B',double),
            ('D_B',double),
            ('D_s',double),            
            ('A_i',double[:]),
            ('y_i',double[:]),
            ('tau_i',double[:]),
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
            ('pension_female',double[:]),
            ('pension_male',double[:]),            
            ('survival',double[:,:,:]),
            ('oap',double[:]),
            ('labor',double[:,:,:,:]),
            ('erp',double[:,:,:,:]),

            # simulation
            ('simM_init',double[:])            

        ]
        
    sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:,:]),
            ('m',double[:]),
            ('v',double[:,:,:,:,:,:]),      

            # post decision
            ('avg_marg_u_plus',double[:,:,:,:,:,:]), 
            ('v_plus_raw',double[:,:,:,:,:,:])                      
        ]     

    simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:]),

            # misc
            ('probs',double[:,:]), 
            ('RA',int32[:]),
            ('euler',double[:,:]),
            ('GovS',double[:,:]),

            # booleans
            ('accuracy',boolean),
            ('tax',boolean),   

            # setup
            ('choiceP',double[:,:,:]),
            ('alive',int32[:,:]),      
            ('labor_pre',double[:,:,:]),
            ('labor_post',double[:,:,:]),
            ('states',int32[:,:])

        ]

    return parlist,sollist,simlist


def couple_lists():

    single_par = single_lists()[0]

    parlist = [ # (name,numba type), parameters, grids etc.

            # preference parameters
            ('pareto_w',double),   
            ('phi_0_male',double),
            ('phi_0_female',double),
            ('phi_1',double),

            # uncertainty/variance parameters
            ('cov',double),

            # grids         
            ('Nxi_men',int32),
            ('Nxi_women',int32),  
            ('xi_corr',double[:,:]),
            ('w_corr',double[:]),            

            # precompute
            ('inc_pens',double[:,:,:,:,:,:]),
            ('inc_mixed',double[:,:,:,:,:,:,:,:,:]),
            ('inc_joint',double[:,:,:,:,:,]),                                 

        ]

    parlist = parlist + single_par   

    sollist = [ # (name, numba type), solution data

            # solution
            ('c',double[:,:,:,:,:,:,:,:]),
            ('m',double[:]),
            ('v',double[:,:,:,:,:,:,:,:]),                     
                
        ]     

    simlist = [ # (name, numba type), simulation data       

            # solution
            ('c',double[:,:]),            
            ('m',double[:,:]),                 
            ('a',double[:,:]),
            ('d',double[:,:,:]),

            # misc
            ('probs',double[:,:,:]), 
            ('RA',int32[:,:]),
            ('euler',double[:,:]),
            ('GovS',double[:,:]),

            # booleans
            ('accuracy',boolean),
            ('tax',boolean),   

            # setup
            ('choiceP',double[:,:,:]), 
            ('alive',int32[:,:,:]),      
            ('shocks_joint',double[:,:,:]),            
            ('labor_pre',double[:,:,:]),
            ('labor_post',double[:,:,:]),
            ('labor_pre_joint',double[:,:]),
            ('labor_post_joint',double[:,:]),
            ('states',int32[:,:])

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

        # ages
        par.oap_age = 65
        par.two_year = 62
        par.erp_age = 60
        
        # oap
        par.B = 61152/par.denom                                 # base rate
        par.y_B = 463500/par.denom                              # maximum annual income before loss of OAP_B
        par.tau_B = 0.3                                         # marginal reduction in deduction regarding income
        par.D_B = 259700/par.denom                              # deduction regarding base value of OAP
        par.D_s = 179400/par.denom                              # maximum deduction in spousal income
        par.A_i = np.array([61560, 28752, 28752])/par.denom     # maximum OAP_A
        par.y_i = np.array([153100, 210800, 306600])/par.denom  # maximum income before loss of OAP_A
        par.tau_i = np.array([0.3, 0.3, 0.15])                  # marginal reduction in OAP_A
        par.D_i = np.array([57300, 115000, 115000])/par.denom   # maximum deduction regarding OAP_A
        
        # erp
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
    if par.couple:
        par.iterator = create_iterator([par.AD,par.ST,par.ST])
        par.ad_min = abs(min(par.AD))
        par.ad_max = max(par.AD)         
    else:
        par.iterator = create_iterator([par.MA,par.ST])       
        par.ad_min = 0
        par.ad_max = 0

def create_iterator(lst):
    indices = 0
    num = len(lst)

    if num == 2:
        iterator = np.zeros((len(lst[0])*len(lst[1]),num),dtype=int)        
        for x in lst[0]:
            for y in range(len(lst[1])):
                iterator[indices] = (x,y)
                indices += 1

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
    if par.couple:                      
        par.xi_corr,par.w_corr = funs.GH_lognorm_corr(par.var,par.cov,par.Nxi_men,par.Nxi_women)    

def init_sim(par,sim):
    """ initialize simulation """
    
    # initialize m and states
    np.random.seed(par.sim_seed)
    state_and_m(par,sim,perc_num=10)

    if par.couple:

        # setup
        init_sim_couple(par,sim)
        
        # draws
        Tr = min(par.simT,par.Tr)        
        mu = -0.5*par.var      
        Cov = np.array(([par.var[0], par.cov], [par.cov, par.var[1]]))      
        shocks_joint = np.exp(np.random.multivariate_normal(mu,Cov,size=(par.simN,min(par.simT,par.Tr))))
        shocks_w = np.exp(np.random.normal(mu[0], np.sqrt(par.var[0]), size=(par.simN,Tr+par.ad_min)))
        shocks_h = np.exp(np.random.normal(mu[1], np.sqrt(par.var[1]), size=(par.simN,Tr+par.ad_min)))            
        
        # income
        init_sim_labor_couple(par,sim,shocks_joint,shocks_w,shocks_h)
    
    else:

        # setup
        init_sim_single(par,sim)
        
        # draws
        shocks = np.nan*np.zeros((par.simN,min(par.simT,par.Tr),2))
        shocks[:,:,0] = np.exp(np.random.normal(-0.5*par.var[0], np.sqrt(par.var[0]), size=(par.simN,min(par.simT,par.Tr))))
        shocks[:,:,1] = np.exp(np.random.normal(-0.5*par.var[1], np.sqrt(par.var[1]), size=(par.simN,min(par.simT,par.Tr))))        
        
        # income
        init_sim_labor_single(par,sim,shocks)        

def init_sim_single(par,sim):
    """ initialize simulation for single model """

    # random draws       
    sim.choiceP = np.random.rand(par.simN,par.simT,1)                            
    deadP = np.random.rand(par.simN,par.simT) 

    # precompute
    MA = sim.states[:,0]
    MAx = np.unique(MA)
    ST = sim.states[:,1]
    STx = np.unique(ST)
    sim.alive = np.ones((par.simN,par.simT),dtype=int)
    alive = sim.alive  
    for t in range(par.simT):

        # if they are dead, they stay dead
        if t > 0:
            alive[alive[:,t-1] == 0,t] = 0

        for ma in MAx:
            for st in STx:

                # indices
                idx = np.nonzero((MA==ma) & (ST==st))[0]

                # alive status (they are all alive first period)
                if t > 0:

                    # alive status
                    pi = transitions.survival_lookup_single(t,ma,st,par)
                    dead = idx[pi < deadP[idx,t]]
                    alive[dead,t] = 0

@njit(parallel=True)
def init_sim_labor_single(par,sim,shocks):
    """ initialize income streams for single model """

    # states
    MA = sim.states[:,0]
    MAx = np.unique(MA)
    ST = sim.states[:,1]
    STx = np.unique(ST)

    # initialize
    np.random.seed(par.sim_seed+1)
    sim.labor_pre = np.nan*np.zeros((par.simN,min(par.simT,par.Tr),2))
    sim.labor_post = np.nan*np.zeros((par.simN,min(par.simT,par.Tr),2))  
    labor_pre = sim.labor_pre
    labor_post = sim.labor_post

    for t in range(par.simT):
        for ma in MAx:
            for st in STx:

                # indices and shocks
                idx = np.nonzero((MA==ma) & (ST==st))[0]

                # labor income
                if t < par.Tr:
                    labor_pre[idx,t,ma] = transitions.labor_pretax(t,ma,st,par)*shocks[idx,t,ma]
                    labor_post[idx,t,ma] = transitions.posttax(t,par,d=1,inc=labor_pre[idx,t,ma],inc_s=np.zeros(len(idx)))

def init_sim_couple(par,sim):
    """ initialize simulation for couple model """

    # random draws for simulation
    ad_min = par.ad_min
    ad_max = par.ad_max
    extend = ad_min + ad_max    
    sim.choiceP = np.random.rand(par.simN,par.simT+extend,2)
    deadP = np.random.rand(par.simN,par.simT+extend,2)  
            
    # precompute
    AD = sim.states[:,0]
    ST_h = sim.states[:,1]
    ST_w = sim.states[:,2]
    ADx = np.unique(AD)
    ST_hx = np.unique(ST_h)
    ST_wx = np.unique(ST_w)

    # 1. alive status
    sim.alive = np.ones((par.simN,par.simT+extend,2),dtype=int)
    alive_w = sim.alive[:,:,0]
    alive_h = sim.alive[:,:,1]
    alive_h[:,-ad_max:] = 0 # last period for men, which we never reach
    deadP_w = deadP[:,:,0]
    deadP_h = deadP[:,:,1]
    for t in range(par.simT):
        if t > 0:
            for ad in ADx:  

                tw_idx = t+ad+par.ad_min
                th_idx = t+par.ad_min
                alive_w[alive_w[:,tw_idx-1] == 0, tw_idx] = 0
                alive_h[alive_h[:,th_idx-1] == 0, th_idx] = 0

                for st_h in ST_hx:
                    for st_w in ST_wx:
                                                
                        pi_h,pi_w = transitions.survival_lookup_couple(t,ad,st_h,st_w,par) 
                        idx = np.nonzero((AD==ad) & (ST_h==st_h) & (ST_w==st_w))[0]                              
                        dead_w = idx[pi_w < deadP_w[idx,tw_idx]]
                        dead_h = idx[pi_h < deadP_h[idx,th_idx]]
                        alive_w[dead_w,tw_idx] = 0
                        alive_h[dead_h,th_idx] = 0

@njit(parallel=True)
def init_sim_labor_couple(par,sim,shocks_joint,shocks_w,shocks_h):
    """ initialize income streams in the couple model """

    # states
    AD = sim.states[:,0]
    ST_h = sim.states[:,1]
    ST_w = sim.states[:,2]
    ADx = np.unique(AD)
    ST_hx = np.unique(ST_h)
    ST_wx = np.unique(ST_w)
    alive_w = sim.alive[:,:,0]
    alive_h = sim.alive[:,:,1]    

    # time
    ad_min = par.ad_min
    ad_max = par.ad_max
    extend = ad_min + ad_max
    Tr = min(par.simT,par.Tr)
    
    # preallocate
    sim.labor_pre = np.nan*np.zeros((par.simN,Tr+extend,2))
    sim.labor_post = np.nan*np.zeros((par.simN,Tr+extend,2))
    sim.labor_pre_joint = np.nan*np.zeros((par.simN,Tr))
    sim.labor_post_joint = np.nan*np.zeros((par.simN,Tr))       
    labor_pre = sim.labor_pre
    labor_post = sim.labor_post
    labor_pre_joint = sim.labor_pre_joint
    labor_post_joint = sim.labor_post_joint

    # individual labor income (post tax can be used for singles, only pre tax can be used for couples, since we don't know pension of spouse) 
    for t in range(Tr+ad_min):
        for ad in ADx:

            t_h = t
            t_w = t+ad
            th_idx = t+ad_min
            tw_idx = t+ad+ad_min

            # husband
            for st_h in np.unique(ST_h):
                if t < par.Tr:      # not forced to retire
                    idx_h = np.nonzero((alive_h[:,th_idx]==1) & (ST_h==st_h))[0]
                    labor_pre[idx_h,th_idx,1] = transitions.labor_pretax(t_h,1,st_h,par)*shocks_h[idx_h,t]
                    labor_post[idx_h,th_idx,1] = transitions.posttax(t_h,par,d=1,inc=labor_pre[idx_h,th_idx,1],inc_s=np.inf*np.ones(len(idx_h)))    # set spouse of inc to infinity so no shared deduction

            # wife
            for st_w in np.unique(ST_w):
                if t+ad < par.Tr:   # not forced to retire
                    idx_w = np.nonzero((alive_w[:,tw_idx]==1) & (ST_w==st_w))[0]
                    labor_pre[idx_w,tw_idx,0] = transitions.labor_pretax(t_w,0,st_w,par)*shocks_w[idx_w,t]
                    labor_post[idx_w,tw_idx,0] = transitions.posttax(t_w,par,d=1,inc=labor_pre[idx_w,tw_idx,0],inc_s=np.inf*np.ones(len(idx_w)))

    # joint labor income (if they both work)
    for t in range(Tr):
        for ad in ADx:
            if t < par.Tr and t+ad < par.Tr:
                for st_h in ST_hx:
                    for st_w in ST_wx:
                        
                        # indices
                        th_idx = t+ad_min
                        tw_idx = t+ad+ad_min                        
                        idx = np.nonzero((alive_h[:,th_idx]==1) & (alive_w[:,tw_idx]==1) & (AD==ad) & (ST_h==st_h) & (ST_w==st_w))[0]
                        
                        # pre tax
                        pre_h = transitions.labor_pretax(t,1,st_h,par)*shocks_joint[idx,t,1]
                        pre_w = transitions.labor_pretax(t+ad,0,st_w,par)*shocks_joint[idx,t,0]
                        labor_pre_joint[idx,t] = pre_h + pre_w
                        
                        # post tax
                        post_h = transitions.posttax(t,par,d=1,inc=pre_h,inc_s=pre_w,d_s=1,t_s=t+ad)
                        post_w = transitions.posttax(t+ad,par,d=1,inc=pre_w,inc_s=pre_h,d_s=1,t_s=t)
                        labor_post_joint[idx,t] = post_h + post_w                        


def state_and_m(par,sim,perc_num=10):
    """ create states and initial wealth (m_init) by loading in relevant information from SASdata"""

    if par.couple:

        # set states
        if par.Thomas:
            data = pd.read_excel('SASdata/Thomas/couple_formue.xlsx')
        else:
            data = pd.read_excel('SASdata/couple_formue.xlsx')
        states = par.iterator
        n_groups = (data['Frac'].to_numpy()*par.simN).astype(int)
        n_groups[-1] = par.simN-np.sum(n_groups[:-1])   # assure it sums to simN
        sim.states = np.transpose(np.vstack((np.repeat(states[:,0],n_groups),
                                             np.repeat(states[:,1],n_groups),
                                             np.repeat(states[:,2],n_groups))))
    
    else:
        
        # set states
        if par.Thomas:
            data = pd.read_excel('SASdata/Thomas/single_formue.xlsx')
        else:
            data = pd.read_excel('SASdata/single_formue.xlsx')
        states = par.iterator
        n_groups = (data['Frac'].to_numpy()*par.simN).astype(int)
        n_groups[-1] = par.simN-np.sum(n_groups[:-1])   # to assure it sums to simN
        sim.states = np.transpose(np.vstack((np.repeat(states[:,0],n_groups),
                                             np.repeat(states[:,1],n_groups))))
        
    # initial liquid wealth
    m_init = np.zeros(len(sim.states))
    idx = np.concatenate((np.zeros(1), np.cumsum(n_groups))).astype(int)
    percentiles = np.linspace(0,100,perc_num+1).astype(int)
    bins = data[list(percentiles)].to_numpy()
    for i in range(n_groups.size):
        m_init[idx[i]:idx[i+1]] = pc_sample(n_groups[i], percentiles, bins[i])
    par.simM_init = m_init

    # add private pension wealth to liquid wealth
    adjust_pension(par,sim)
    sim.m = np.nan*np.zeros((par.simN,par.simT))
    sim.m[:,0] = m_init
    states = sim.states

    if par.couple:
        for st_h in range(len(par.ST)):
            for st_w in range(len(par.ST)):
                idx = np.nonzero((states[:,1]==st_h) & (states[:,2]==st_w))[0]
                hs_h = transitions.state_translate(st_h,'high_skilled',par)
                hs_w = transitions.state_translate(st_w,'high_skilled',par)
                sim.m[idx,0] += (1-par.IRA_tax)*(par.pension_male[hs_h] + par.pension_female[hs_w])

    else:
        for ma in par.MA:
            for st in range(len(par.ST)):
                idx = np.nonzero((states[:,0]==ma) & (states[:,1]==st))[0]
                hs = transitions.state_translate(st,'high_skilled',par)
                if ma == 0:
                    sim.m[idx,0] += (1-par.IRA_tax)*par.pension_female[hs]
                elif ma == 1:
                    sim.m[idx,0] += (1-par.IRA_tax)*par.pension_male[hs]
        
def pc_sample(N,percentiles,bins):
    """ N samples from a dsitribution given its percentiles and bins (assumes equal spacing between percentiles)"""
    diff = np.diff(percentiles)
    assert np.allclose(diff[0],diff)
    n = int(N/diff.size)
    draws = np.random.uniform(low=bins[:-1], high=bins[1:], size=(n,diff.size)).ravel()
    return np.concatenate((draws, np.random.uniform(low=bins[0], high=bins[-1], size=(N-n*diff.size)))) # to assure we return N samples                        

def adjust_pension(par,sim):

    # unpack 
    states = sim.states
    priv_pension = np.array([par.priv_pension_female, par.priv_pension_male])
    
    # adjust pension
    for ma in par.MA:

        # indices
        idx_low = np.nonzero((states[:,0]==ma) & (np.isin(states[:,1], (0,2))))[0]
        idx_high = np.nonzero((states[:,0]==ma) & (np.isin(states[:,1], (1,3))))[0]            
            
        # adjust private pension
        share = len(idx_high) / (len(idx_low) + len(idx_high))          # share of high skilled
        pens_low = Xlow(par.g_adjust,share)*priv_pension[ma]
        pens_high = Xhigh(par.g_adjust,share)*priv_pension[ma]
        assert np.allclose(share*pens_high + (1-share)*pens_low, priv_pension[ma])

        # store
        if ma == 0:
            par.pension_female = np.array((pens_low, pens_high))

        elif ma == 1:
            par.pension_male = np.array((pens_low, pens_high))
    
def Xlow(g,share):
    return 1/(1+share*g)

def Xhigh(g,share):
    return (1+g)/(1+share*g)    