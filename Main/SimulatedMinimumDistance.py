#####################################################################
# Source: https://www.dropbox.com/s/g1im7uqzukvqo53/web_sens.zip?dl=0
# Thanks to Thomas Jørgensen for sharing the code
#####################################################################

# global modules
import numpy as np
import time
import scipy as sci
from scipy.optimize import minimize
import pickle
import itertools
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

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

    def __init__(self,model,mom_data,mom_fun,recompute=False,bounds=None,name='baseline',method='nelder-mead',est_par=[],par_save={},options={'disp': False},print_iter=[False,1],save=False,**kwargs): # called when created
        
        # settings for model
        self.model = model
        self.mom_data = mom_data
        self.mom_fun = mom_fun
        self.recompute = recompute
        self.name = name

        # settings for estimation
        self.bounds = bounds
        self.options = options
        self.method = method
        self.est_par = est_par

        # settings for printing and saving
        self.save = save
        self.par_save = par_save     
        self.obj_save = []   
        self.print_iter = print_iter
        self.iter = 0
        self.time = {self.iter: time.time()}

    def obj_fun(self,theta,W,*args):
        
        # print parameters
        if self.print_iter[0]:
            self.iter += 1
            if self.iter % self.print_iter[1] == 0:
                self.time[self.iter] = time.time()
                toctic = self.time[self.iter] - self.time[self.iter-self.print_iter[1]]
                print('Iteration:', self.iter, '(' + str(np.round(toctic/60,2)) + ' minutes)')
                for p in range(len(theta)):
                    print(f' {self.est_par[p]}={theta[p]:2.4f}', end='')

        # hardcode constraint on variance
        if 'sigma_eta' in self.est_par and theta[self.est_par.index('sigma_eta')] < 0:
            self.obj = np.inf
        else:

            # 1. update parameters 
            for i in range(len(self.est_par)):
                setattr(self.model.par,self.est_par[i],theta[i]) # like par.key = val
                if self.model.couple and hasattr(self.model.Single.par,self.est_par[i]):
                    setattr(self.model.Single.par,self.est_par[i],theta[i]) # update also in nested single model                

            # update of phi0 - just uncomment this, when estimating both
            if 'phi_0_male' in self.est_par:
                idx = self.est_par.index('phi_0_male')
                setattr(self.model.par,'phi_0_female',theta[idx])

            elif 'phi_0_female' in self.est_par:
                idx = self.est_par.index('phi_0_female')
                setattr(self.model.par,'phi_0_male',theta[idx])

            # 2. solve model with current parameters
            self.model.solve(recompute=self.recompute)

            # 3. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
            self.model.simulate()
            self.mom_sim = self.mom_fun(self.model,*args)

            # 4. calculate objective function and return it
            diff = self.mom_data - self.mom_sim
            self.obj  = ((np.transpose(diff) @ W) @ diff)

        # print obj
        if self.print_iter[0]:
            if self.iter % self.print_iter[1] == 0:
                print(f' -> {self.obj:2.4f}')

        # save
        if self.save:
            for p in range(len(theta)):
                self.par_save[self.est_par[p]].append(theta[p])
            self.obj_save.append(self.obj)                
                    
        # return
        return self.obj 

    def estimate(self,theta0,W,*args):
        # TODO: consider multistart-loop with several algortihms - that could alternatively be hard-coded outside
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        # estimate
        self.est_out = minimize(self.obj_fun, theta0, (W, *args), bounds=self.bounds, method=self.method,options=self.options)

        # return output
        self.est = self.est_out.x
        self.W = W     
    
    def MultiStart(self,theta0,weight,options={'print': True, 'time': 'min', 'finalN': int(5e5)}):
            
        # time
        tic_total = time.time()
            
        # preallocate
        theta = np.nan*np.zeros(np.array(theta0).shape)
        obj = np.nan*np.zeros(len(theta0))

        # options
        self.options['xatol'] = 0.001
        self.options['fatol'] = 0.001        
            
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

        # change settings
        self.options['xatol'] = 0.0001
        self.options['fatol'] = 0.0001
        startN = self.model.par.simN
        self.model.par.simN = options['finalN']
        self.model.recompute()

        # estimate
        idx = np.argmin(obj)
        self.estimate(theta[idx],weight)
        toc_total = time.time()
        
        # prints
        if options['print']:
            print('final estimation:')
            print('success:', self.est_out.success,'|', 'feval:', self.est_out.nfev, '|', 'obj:', self.obj)
            print('total estimation time:', str(np.round((toc_total-tic_total)/(60**2),1)) + ' hours')
            print('start par:', theta[idx])            
            print('par:', self.est)
            print('')

        # reset N
        self.model.par.simN = startN
        self.model.recompute()

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

        # reset parameters
        for p in range(len(self.est_par)):
            setattr(self.model.par,self.est_par[p],theta[p])

        # DO my suggestion
        if fixed_par_str:
            # mine: calculate the numerical gradient wrt parameters in fixed_par

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

            # reset parameters
            for p in range(len(self.est_par)):
                setattr(self.model.par,self.est_par[p],gamma[p])
            self.est_par = est_par

            # sensitivity
            self.sens2 = Lambda @ grad_g
            ela = np.empty((len(theta),len(gamma)))
            semi_ela = np.empty((len(theta),len(gamma)))
            for t in range(len(theta)):
                for g in range(len(gamma)):
                    ela[t,g] = self.sens2[t,g]*gamma[g]/theta[t]    
                    semi_ela[t,g] = self.sens2[t,g]/theta[t]

            self.sens2e = ela
            self.sens2semi = semi_ela

def ols(y,X):
    return np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y

def prepareSingle_reg(model,ma):
    
    # info
    idx = np.nonzero(model.sim.states[:,0]==ma)[0]
    alive = model.sim.alive[idx,3:]
    hs = np.isin(model.sim.states[:,1],[1,3])[idx]
    Nt = np.sum(alive,axis=0)
    T = alive.shape[1]
    N = np.sum(Nt)
    age = np.concatenate((np.zeros(1), np.cumsum(Nt))).astype(int)    

    # initialize y
    y = np.zeros(N)

    # create X
    X = np.zeros((N,T+1))
    for i in range(len(age)-1):
        X[age[i]:age[i+1],i] = 1
        X[age[i]:age[i+1],-1] = hs[alive[:,i]==1]*1

    return {'y': y, 'X': X, 'alive': alive, 'idx': idx, 'age': age}

def MomFunSingle_reg(model,pre):
    
    # retirement age for all
    ret_age_total = np.nanargmin(model.sim.d,axis=1)+57
    y_lst = []
    for ma in [0,1]:
        
        # unpack
        y = pre[ma]['y']
        X = pre[ma]['X']
        alive = pre[ma]['alive']
        idx = pre[ma]['idx']
        age = pre[ma]['age']
        ret_age = ret_age_total[idx]
        
        # create y
        for i in range(len(age)-1):
            y[age[i]:age[i+1]] = ret_age[alive[:,i]==1]==i+60
            
        # ols
        y_lst.append(ols(y,X))
        
    return np.concatenate(y_lst)


def MomFunSingle(model,calc='mean'):
    """ compute moments for single model """

    # unpack
    sim = model.sim
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
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)                
            if calc == 'mean': 
                mom[:,i] = np.nanmean(probs[idx,:],axis=0)
            elif calc == 'std':
                mom[:,i] = np.nanstd(probs[idx,:],axis=0)
    return mom.ravel() # collapse across rows (C-order)

def MomFunSingleThomas(model,calc='mean'):
    """ compute aggregate moments for matching with data from Thomas """

    # unpack
    sim = model.sim
    MA = sim.states[:,0]    
    probs = sim.probs[:,1:] # 1: means exclude age 57 (since first prob is at 58)
        
    # initialize
    mom = np.zeros((2,probs.shape[1]))
    
    # compute moments
    for ma in [0,1]:
        idx = np.nonzero(MA==ma)[0]            
        if calc == 'mean': 
            mom[ma] = np.nanmean(probs[idx,:],axis=0)
        elif calc == 'std':
            mom[ma] = np.nanstd(probs[idx,:],axis=0)
    return mom.ravel() # collapse across rows (C-order)    

def MomFunCoupleThomas(model,calc='mean',ages=[58,68]):
    """ compute aggregate moments for matching with data from Thomas """

    # unpack
    sim = model.sim
    par = model.par

    # extract probs
    x = np.arange(ages[0], ages[1]+1)    
    probs_h = sim.probs[:,transitions.inv_age(x,par)+par.ad_min,1]
    probs_w = sim.probs[:,transitions.inv_age(x,par)+par.ad_min,0]     
        
    # initialize
    mom = np.zeros((2,len(x)))
    
    # compute moments
    if calc == 'mean':
        mom[1] = np.nanmean(probs_h,axis=0)
        mom[0] = np.nanmean(probs_w,axis=0)
    elif calc == 'std':
        mom[1] = np.nanstd(probs_h,axis=0)
        mom[0] = np.nanstd(probs_w,axis=0)

    return mom.ravel() # collapse across rows (C-order)       

# def MomFunCouple(model,calc='mean',ages=[58,68]):
#     """ compute moments for couple model """    
    
#     # unpack
#     par = model.par
#     sim = model.sim
#     AD = sim.states[:,0]
#     ADx = np.unique(AD)
#     ST_h = sim.states[:,1]    
#     ST_w = sim.states[:,2]
#     x = np.arange(ages[0], ages[1]+1)    
#     probs_h = sim.probs[:,transitions.inv_age(x,par)+par.ad_min,1]
#     probs_w = sim.probs[:,transitions.inv_age(x,par)+par.ad_min,0]    

#     # initialize
#     T = len(x)
#     N = len(ADx)+4
#     mom = np.zeros((2,T,N))

#     # 1. across AD
#     for i in range(len(ADx)):
#         ad = ADx[i]
#         idx = np.nonzero((AD==ad))[0]
#         with warnings.catch_warnings(): # ignore this specific warning
#             warnings.simplefilter("ignore", category=RuntimeWarning)        
#             if calc == 'mean':
#                 mom[0,:,i] = np.nanmean(probs_h[idx],axis=0)
#                 mom[1,:,i] = np.nanmean(probs_w[idx],axis=0)
#             elif calc == 'std':
#                 mom[0,:,i] = np.nanstd(probs_h[idx],axis=0)
#                 mom[1,:,i] = np.nanstd(probs_w[idx],axis=0)     

#     # 2. across education
#     # men
#     hs_m = np.isin(ST_h,[1,3])
#     mom[0,:,i+1] = np.nanmean(probs_h[~hs_m],axis=0)
#     mom[0,:,i+2] = np.nanmean(probs_h[hs_m],axis=0)

#     # women
#     hs_w = np.isin(ST_w,[1,3])
#     mom[1,:,i+1] = np.nanmean(probs_w[~hs_w],axis=0)
#     mom[1,:,i+2] = np.nanmean(probs_w[hs_w],axis=0)

#     # 3. across elig
#     # men
#     elig_m = np.isin(ST_h,[2,3])
#     mom[0,:,i+3] = np.nanmean(probs_h[~elig_m],axis=0)
#     mom[0,:,i+4] = np.nanmean(probs_h[elig_m],axis=0)

#     # women
#     elig_w = np.isin(ST_w,[2,3])
#     mom[1,:,i+3] = np.nanmean(probs_w[~elig_w],axis=0)
#     mom[1,:,i+4] = np.nanmean(probs_w[elig_w],axis=0)                        

#     # return
#     mom = mom.ravel()
#     mom[np.isnan(mom)] = 0  # set nan to zero
#     return mom    

# # Moments on retirement status of spouse
# def MomFunCouple(model,calc='mean',ages=[58,68]):
#     """ compute moments for couple model """    
    
#     # unpack
#     sim = model.sim
#     par = model.par
#     ST_h = sim.states[:,1]    
#     ST_w = sim.states[:,2]
#     iterator = np.array(list(itertools.product([0, 1, 2, 3], repeat=2)))
#     x = np.arange(ages[0], ages[1]+1)    
#     x_idx = transitions.inv_age(x,par)+par.ad_min
#     probs_h = sim.probs[:,x_idx,1]
#     probs_w = sim.probs[:,x_idx,0]    
#     sret_h = sim.spouse_ret[:,x_idx,1]
#     sret_w = sim.spouse_ret[:,x_idx,0]

#     # initialize
#     T = len(x)
#     N = len(iterator)
#     mom = np.zeros((2,2,T,N))

#     for i in range(len(iterator)):
#         st_h = iterator[i,0]
#         st_w = iterator[i,1]
#         idx = np.nonzero((ST_h==st_h) & (ST_w==st_w))[0]

#         if calc == 'mean':

#             # men
#             mom[0,0,:,i] = np.nanmean(probs_h[idx]*sret_h[idx],axis=0)
#             mom[0,1,:,i] = np.nanmean(probs_h[idx]*(1-sret_h[idx]),axis=0)

#             # women
#             mom[1,0,:,i] = np.nanmean(probs_w[idx]*sret_w[idx],axis=0)
#             mom[1,1,:,i] = np.nanmean(probs_w[idx]*(1-sret_w[idx]),axis=0)            

#         elif calc == 'std':

#             # men
#             mom[0,0,:,i] = np.nanstd(probs_h[idx]*sret_h[idx],axis=0)
#             mom[0,1,:,i] = np.nanstd(probs_h[idx]*(1-sret_h[idx]),axis=0)

#             # women
#             mom[1,0,:,i] = np.nanstd(probs_w[idx]*sret_w[idx],axis=0)
#             mom[1,1,:,i] = np.nanstd(probs_w[idx]*(1-sret_w[idx]),axis=0)            

#     # return
#     mom = mom.ravel()
#     mom[np.isnan(mom)] = 0  # set nan to zero
#     return mom

def MomFunCouple(model,calc='mean',ages=[58,68]):
    """ compute moments for couple model """    
    
    # unpack
    sim = model.sim
    par = model.par

    # unpack states
    states = sim.states
    AD = states[:,0]
    ADx = np.unique(AD)
    ST_h = sim.states[:,1]    
    ST_w = sim.states[:,2]
    iterator = np.array(list(itertools.product([0, 1, 2, 3], repeat=2)))
    
    # extract probs
    x = np.arange(ages[0], ages[1]+1)    
    probs_h = sim.probs[:,transitions.inv_age(x,par)+par.ad_min,1]
    probs_w = sim.probs[:,transitions.inv_age(x,par)+par.ad_min,0]    

    # initialize
    T = len(x)
    N = len(ADx)+len(iterator)
    mom = np.zeros((2,T,N))

    # 1. across AD
    for i in range(len(ADx)):
        ad = ADx[i]
        idx = np.nonzero((AD==ad))[0]
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)        
            if calc == 'mean':
                mom[0,:,i] = np.nanmean(probs_h[idx],axis=0)
                mom[1,:,i] = np.nanmean(probs_w[idx],axis=0)
            elif calc == 'std':
                mom[0,:,i] = np.nanstd(probs_h[idx],axis=0)
                mom[1,:,i] = np.nanstd(probs_w[idx],axis=0)            

    # 2. across couple states
    for i in range(len(iterator)):
        st_h = iterator[i,0]
        st_w = iterator[i,1]
        idx = np.nonzero((ST_h==st_h) & (ST_w==st_w))[0]
        j = i + len(ADx)
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)            
            if calc == 'mean':
                mom[0,:,j] = np.nanmean(probs_h[idx],axis=0)
                mom[1,:,j] = np.nanmean(probs_w[idx],axis=0)
            elif calc == 'std':
                mom[0,:,j] = np.nanstd(probs_h[idx],axis=0)
                mom[1,:,j] = np.nanstd(probs_w[idx],axis=0)                          

    # return
    mom = mom.ravel()
    mom[np.isnan(mom)] = 0  # set nan to zero
    return mom

def weight_matrix_single(std,shape,factor=[1]*11):
    
    # preallocate
    std_inv = np.zeros(std.shape)
    
    # find all above zero
    idx = np.nonzero(std>0)[0]
    
    # invert
    std_inv[idx] = 1/std[idx]

    # scale weights
    y_all = std_inv.reshape(shape).copy()
    for t in range(len(factor)):
        y_all[t] = y_all[t]*factor[t]

    # weight matrix
    return np.eye(y_all.size)*y_all.ravel()

def weight_matrix_couple(std,shape,factor=[1]*11):

    # preallocate
    std[np.isnan(std)] = 0
    std_inv = np.zeros(std.shape)

    # find all above zero
    idx = np.nonzero(std>0)[0]

    # invert
    std_inv[idx] = 1/std[idx]

    # scale weights    
    y_all = std_inv.reshape(shape).copy()    
    for t in range(len(factor)):
        y_all[:,t] = y_all[:,t]*factor[t]

    # weight matrix
    return np.eye(y_all.size)*y_all.ravel()

def start(N,bounds):
    outer = []
    for _ in range(N):
        inner = []
        for j in range(len(bounds)):
            inner.append(np.round(np.random.uniform(bounds[j][0],bounds[j][1]),3))
        outer.append(inner)
    return outer    

def identification(model,true_par,est_par,true_save,par_save,par_latex,start,end,N,plot=True,save_plot=True):
    
    # update parameters
    for i in range(len(est_par)):
        setattr(model.par, est_par[i], true_par[i])
        if model.couple and hasattr(model.Single.par,est_par[i]):
            setattr(model.Single.par,est_par[i],true_par[i])            
    
    # data
    model.solve()
    model.simulate()
    def mom_fun(model):
        return MomFunCouple(model)    
    mom_data = mom_fun(model)
    weight = np.eye(mom_data.size)
    
    # grids
    x1 = np.linspace(start[0],end[0],N)
    x2 = np.linspace(start[0],end[0],N)
    # a = true_par[0]
    # b = true_par[1]
    # Q = a*true_par[2] + b*true_par[3]
    # x2 = np.linspace(1,2,5)
    # x1 = (1/a)*(Q-b*x2)    
    x1,x2 = np.meshgrid(x1,x2)
    x1,x2 = x1.ravel(),x2.ravel()
    
    # estimate
    smd = SimulatedMinimumDistance(model,mom_data,mom_fun,save=True)
    smd.est_par = par_save
    smd.par_save = {par_save[0]: [], par_save[1]: []}
    for i in range(N*N):
        print(i, end=' ')    # track progress because it takes so long time
        theta = [x1[i],x2[i]]
        smd.obj_fun(theta,weight)
    
    # reset parameters
    for i in range(len(est_par)):
        setattr(model.par, est_par[i], true_par[i])
        if model.couple and hasattr(model.Single.par,est_par[i]):
            setattr(model.Single.par,est_par[i],true_par[i])                
    
    # return
    x1 = x1.reshape(N,N) 
    x2 = x2.reshape(N,N)
    y = np.array(smd.obj_save).reshape(N,N)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1,x2,y, 
                        rstride=2, cstride=2,
                        cmap=plt.cm.jet,
                        alpha=0.7,
                        linewidth=0.25)
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.set_xlabel(par_latex[0], fontsize=20)
        ax.set_ylabel(par_latex[1], fontsize=20)
        ax.set_xticklabels(['',np.round(np.min(x1),1),'','','','',np.round(np.max(x1),1)])
        ax.set_yticklabels(['',np.round(np.min(x2),1),'','','','',np.round(np.max(x2),1)])
        ax.tick_params(axis='both', which='major', labelsize=12)  
        fig.tight_layout()
        if save_plot:
            return fig        
    else:
        return x1,x2,y

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
