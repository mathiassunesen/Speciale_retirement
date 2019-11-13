
# global modules
import numpy as np
import time
import scipy as sci
from scipy.optimize import minimize
import pickle

# local modules
import transitions

# TODO: 
# 1) add a saving-module?:
# 2) multistart-loop?
class SimulatedMinimumDistance():
    ''' 
    This class performs simulated minimum distance (self) estimation.
    Input requirements are
    - model: Class with solution and simulation capabilities: model.solve() and model.simulate(). 
             Properties of model should be contained in model.par
    - mom_data: np.array (1d) of moments in the data to be used for estimation
    - mom_fun: function used to calculate moments in simulated data. Should return a 1d np.array
    
    '''    

    def __init__(self,model,mom_data,mom_fun,recompute=False,bounds=None,name='baseline',method='nelder-mead',est_par=[],options={'disp': False},print_iter=[False,1],save=False,**kwargs): # called when created
        
        self.model = model
        self.mom_data = mom_data
        self.mom_fun = mom_fun
        self.recompute = recompute
        self.name = name

        # estimation settings
        self.bounds = bounds
        self.options = options
        self.print_iter = print_iter
        self.method = method
        self.iter = 0
        self.time = {self.iter: time.time()}

    def obj_fun(self,theta,W,*args):
        
        self.iter += 1

        if self.print_iter[0]:
            if self.iter % self.print_iter[1] == 0:
                self.time[self.iter] = time.time()
                toctic = self.time[self.iter] - self.time[self.iter-self.print_iter[1]]
                print('Iteration:', self.iter, '(' + str(np.round(toctic/60,2)) + ' minutes)')
                for p in range(len(theta)):
                    print(f' {self.est_par[p]}={theta[p]:2.4f}', end='')

        # 1. update parameters 
        for i in range(len(self.est_par)):
            setattr(self.model.par,self.est_par[i],theta[i]) # like par.key = val
            if self.model.couple and hasattr(self.model.Single.par,self.est_par[i]):
                setattr(self.model.Single.par,self.est_par[i],theta[i]) # update also in nested single model                

        # 2. solve model with current parameters
        if self.recompute:
            self.model.recompute()
        self.model.solve()

        # 3. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
        self.model.simulate()
        self.mom_sim = self.mom_fun(self.model.sim,*args)

        # 4. calculate objective function and return it
        diff = self.mom_data - self.mom_sim
        self.obj  = ((np.transpose(diff) @ W) @ diff)

        if self.print_iter[0]:
            if self.iter % self.print_iter[1] == 0:
                if self.model.couple:
                    print(f' -> {self.obj:2.4f}')
                else:
                    print(f' -> {self.obj:2.4f}')
        
        return self.obj 

    def estimate(self,theta0,W,*args):
        # TODO: consider multistart-loop with several algortihms - that could alternatively be hard-coded outside
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        # estimate
        self.est_out = minimize(self.obj_fun, theta0, (W, *args), bounds=self.bounds, method=self.method,options=self.options)

        # return output
        self.est = self.est_out.x
        self.W = W     
    
    def MultiStart(self,theta0,weight,options={'print': True, 'time': 'min'}):
            
        # time
        tic_total = time.time()
            
        # preallocate
        theta = np.nan*np.zeros(np.array(theta0).shape)
        obj = np.nan*np.zeros(len(theta0))
            
        for p in range(len(theta0)):
                
            # estimate
            tic = time.time()
            self.estimate(theta0[p],weight)
            toc = time.time()
                
            # save
            theta[p] = self.est
            obj[p] = self.obj
                
            # print
            if options['print']:
                    
                if options['time'] == 'sec':
                    tid = str(np.round(toc-tic,1)) + ' sec'
                if options['time'] == 'min':
                    tid = str(np.round((toc-tic)/60,1)) + ' min'
                if options['time'] == 'hours':
                    tid = str(np.round((toc-tic)/(60**2),1)) + ' hours'
                    
                print(p+1, 'estimation:')
                print('success:', self.est_out.success,'|', 'feval:', self.est_out.nfev, '|', 
                      'time:', tid, '|', 'obj:', self.obj)
                print('start par:', theta0[p])
                print('par:      ', self.est)
                print('')
                    
        # final estimation
        idx = np.argmin(obj)
        self.estimate(theta[idx],weight)
        toc_total = time.time()
        if options['print']:
            print('final estimation:')
            print('success:', self.est_out.success,'|', 'feval:', self.est_out.nfev, '|', 'obj:', self.obj)
            print('total estimation time:', str(np.round((toc_total-tic_total)/(60**2),1)) + ' hours')
            print('start par:', theta[idx])            
            print('par:', self.est)
            print('')

    def std_error(self,theta,W,Nobs,Nsim,step=1.0e-4,*args):
        ''' Calculate standard errors and sensitivity measures '''

        num_par = len(theta)
        num_mom = len(W[0])

        self.obj_fun(theta,W)
        diff = self.mom_data - self.mom_sim
        diff = diff.reshape((len(diff),1))
        Omega = diff @ np.transpose(diff)
        
        # 1. numerical gradient. The objective function is (data - sim)'*W*(data - sim) so take the negative of mom_sim
        grad = np.empty((num_mom,num_par))
        for p in range(num_par):
            theta_now = theta[:] 

            step_now  = np.zeros(num_par)
            step_now[p] = np.fmax(step,step*theta_now[p])

            self.obj_fun(theta_now + step_now,W,*args)
            mom_forward = - self.mom_sim

            self.obj_fun(theta_now - step_now,W,*args)
            mom_backward = - self.mom_sim

            grad[:,p] = (mom_forward - mom_backward)/(2.0*step_now[p])

        # 2. asymptotic standard errors [using Omega: V(mom_data_i). If bootstrapped, remember to multiply by Nobs]
        GW  = np.transpose(grad) @ W
        GWG = GW @ grad

        Avar = np.linalg.inv(GWG) @ ( GW @ Omega @ np.transpose(GW) ) @ np.linalg.inv(GWG)
        fac  = (1.0 + 1.0/Nsim)/Nobs # Nsim: number of simulated observations, Nobs: number of observations in data
        self.std = np.sqrt( fac*np.diag(Avar) )

        # 3. Sensitivity measures
        self.sens1 = - np.linalg.inv(GWG) @ GW  # Andrews I, Gentzkow M, Shapiro JM: "Measuring the Sensitivity of Parameter Estimates to Estimation Moments." Quarterly Journal of Economics. 2017;132 (4) :1553-1592
       
    def sensitivity(self,theta,W,fixed_par_str=None,step=1.0e-4,*args):
        ''' sensitivity measures '''

        num_par = len(theta)
        num_mom = len(W[0])

        # 1. numerical gradient. The objective function is (data - sim)'*W*(data - sim) so take the negative of mom_sim
        grad = np.empty((num_mom,num_par))
        for p in range(num_par):
            theta_now = theta[:] 

            step_now    = np.zeros(num_par)
            step_now[p] = np.fmax(step,step*theta_now[p])

            self.obj_fun(theta_now + step_now,W,*args)
            mom_forward = - self.mom_sim

            self.obj_fun(theta_now - step_now,W,*args)
            mom_backward = - self.mom_sim

            grad[:,p] = (mom_forward - mom_backward)/(2.0*step_now[p])

        # 2. Sensitivity measures
        GW  = np.transpose(grad) @ W
        GWG = GW @ grad
        Lambda = - np.linalg.inv(GWG) @ GW

        # 3. Sensitivity measures
        self.sens1 = Lambda  # Andrews I, Gentzkow M, Shapiro JM: "Measuring the Sensitivity of Parameter Estimates to Estimation Moments." Quarterly Journal of Economics. 2017;132 (4) :1553-1592

        # DO my suggestion
        if fixed_par_str:
            # mine: calculate the numerical gradient wrt parameters in fixed_par
            # update parameters
            for p in range(len(self.est_par)):
                setattr(self.model.par,self.est_par[p],theta[p])

            # change the estimation parameters to be the fixed ones
            est_par = self.est_par
            self.est_par = fixed_par_str

            # construct vector of fixed values
            gamma = np.empty(len(self.est_par))
            for p in range(len(self.est_par)):
                gamma[p] = getattr(self.model.par,self.est_par[p])

            # calculate gradient with respect to gamma
            num_gamma = len(gamma)
            grad_g = np.empty((num_mom,num_gamma))
            for p in range(num_gamma):
                gamma_now = gamma[:] 

                step_now    = np.zeros(num_gamma)
                step_now[p] = np.fmax(step,step*gamma_now[p])

                self.obj_fun(gamma_now + step_now,W,*args)
                mom_forward = - self.mom_sim

                self.obj_fun(gamma_now - step_now,W,*args)
                mom_backward = - self.mom_sim

                grad_g[:,p] = (mom_forward - mom_backward)/(2.0*step_now[p])

            self.est_par = est_par
            self.sens2 = Lambda @ grad_g

            ela = np.empty((len(theta),len(gamma)))
            for t in range(len(theta)):
                for g in range(len(gamma)):
                    ela[t,g] = self.sens2[t,g]*gamma[g]/theta[t]    
            
            self.sens2e = ela

def MomFunSingle(sim,par,calc='mean'):
    """ compute moments for single model """

    # unpack
    states = np.unique(sim.states,axis=0)
    MA = sim.states[:,0]
    ST = sim.states[:,1]    
    probs = sim.probs[:,1:] # 1: means exclude age 57 (since first prob is at 58)
        
    # initialize
    T = probs.shape[1]
    N = len(states)
    mom = np.zeros((T,N))
    
    # compute moments
    for i in range(N):
        ma = states[i,0]
        st = states[i,1]
        idx = np.nonzero((MA==ma) & (ST==st))[0]
        if calc == 'mean': 
            mom[:,i] = np.nanmean(probs[idx,:],axis=0)
        elif calc == 'std':
            mom[:,i] = np.nanstd(probs[idx,:],axis=0)
    return mom.ravel() # collapse across rows (C-order)

def MomFunCouple(sim,par,calc='mean'):
    """ compute moments for couple model """    
    
    # unpack
    states = sim.states
    AD = states[:,0]
    ST_h = sim.states[:,1]    
    ST_w = sim.states[:,2]
    ADx = np.unique(AD)
    STx = np.unique(sim.states[1:],axis=0) # exclude AD at first entry
    probs_h = sim.probs[:,1+par.ad_min+57-par.start_T:,1]
    probs_w = sim.probs[:,1+par.ad_min+57-par.start_T:,0]
    
    # initialize
    T = probs_h.shape[1]    
    N = len(ADx)+len(STx)
    mom = np.zeros((2,T,N))

    # 1. across AD
    for i in range(len(ADx)):
        ad = ADx[i]
        idx = np.nonzero((AD==ad))[0]
        if calc == 'mean':
            mom[0,:,i] = np.nanmean(probs_h[idx],axis=0)
            mom[1,:,i] = np.nanmean(probs_w[idx],axis=0)
        elif calc == 'std':
            mom[0,:,i] = np.nanstd(probs_h[idx],axis=0)
            mom[1,:,i] = np.nanstd(probs_w[idx],axis=0)            

    # 2. across couple states
    for i in range(len(STx)):
        st_h = states[i,1]
        st_w = states[i,2]
        idx = np.nonzero((ST_h==st_h) & (ST_w==st_w))[0]
        j = i + len(ADx)
        if calc == 'mean':
            mom[0,:,j] = np.nanmean(probs_h[idx],axis=0)
            mom[1,:,j] = np.nanmean(probs_w[idx],axis=0)
        elif calc == 'std':
            mom[0,:,j] = np.nanstd(probs_h[idx],axis=0)
            mom[1,:,j] = np.nanstd(probs_w[idx],axis=0)                          

    # return
    return mom.ravel()

# def MomFunCouple(sim,par,std=False):
#     """ compute moments for couple model """    
    
#     # unpack
#     states = np.unique(sim.states,axis=0)
#     AD = sim.states[:,0]
#     ST_h = sim.states[:,1]    
#     ST_w = sim.states[:,2]
#     probs_h = sim.probs[:,1+par.ad_min:,1] # 1: means exclude age 57 (since first prob is at 58)
#     probs_w = sim.probs[:,1+par.ad_min:,0]
    
#     # initialize
#     T = probs_h.shape[1]
#     N = len(states)
#     mom = np.zeros((2,T,N))
    
#     # compute moments
#     for i in range(N):
#         ad = states[i,0]
#         st_h = states[i,1]
#         st_w = states[i,2]
#         idx = np.nonzero((AD==ad) & (ST_h==st_h) & (ST_w==st_w))[0]
#         if std:
#             mom[0,:,i] = np.nanstd(probs_h[idx,:],axis=0)
#             mom[1,:,i] = np.nanstd(probs_w[idx,:],axis=0)           
#         else:
#             mom[0,:,i] = np.nanmean(probs_h[idx,:],axis=0)
#             mom[1,:,i] = np.nanmean(probs_w[idx,:],axis=0)            
#     mom = mom.ravel()
#     mom[np.isnan(mom)] = 0 # there will be some nans,since we don't observe probs for all wives (due to age difference)
#     return mom             # and some groups are potentially very small, so if all die, then we don't observe probs

def save_est(est_par,theta,name):
    """ save estimated parameters to "estimates"-folder """
    EstDict = dict(zip(est_par,theta))
    with open('estimates/'+str(name)+'.pickle', 'wb') as handle:
        pickle.dump(EstDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_est(name,couple=False):
    """ load estimated parameters from "estimates"-folder """    
    with open('estimates/'+str(name)+'.pickle', 'rb') as handle:
        EstDict = pickle.load(handle)
    
    if couple:
        single_par = ['alpha_0_male', 'alpha_0_female', 'alpha_1', 'sigma_eta']
        CoupleDict = {}
        SingleDict = {}
        for key,val in EstDict.items():
            CoupleDict[key] = val
            if key in single_par:
                SingleDict[key] = val
        return CoupleDict,SingleDict
    else:
        return EstDict
