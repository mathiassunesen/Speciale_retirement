# global modules
from numba import boolean, int32, int64, float64, double, njit, prange, typeof

def lists(couple):

    single_parlist = [ # (name,numba type), parameters, grids etc.

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

    couple_parlist = [ # (name,numba type), parameters, grids etc.

            # preference parameters
            ('pareto_w',double),
            ('v',double),            
            ('phi_0_male',double),
            ('phi_0_female',double),
            ('phi_1',double),
                
            # uncertainty/variance parameters
            ('sigma_eta',double), 
            ('sigma_xi_men',double), 
            ('sigma_xi_women',double),
            ('sigma_xi_corr',double)            
        ]

    couple_parlist = single_parlist + couple_parlist
        
    singe_sollist = [ # (name, numba type), solution data

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

    couple_sollist = [ # (name, numba type), solution data

            # solution
            ('c_single',double[:,:,:,:,:]),
            ('m_single',double[:,:,:,:,:]),
            ('v_single',double[:,:,:,:,:])                               
        ]       

    couple_sollist = single_sollist + couple_sollist

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

    if couple:
        return common_par+couple_par,common_sol+couple_sol,simlist
    else:
        return common_par,common_sol,simlist