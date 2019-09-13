import numpy as np
import scipy as sci
from scipy.optimize import minimize

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

    def __init__(self,model,mom_data,mom_fun,name='baseline',method='nelder-mead',est_par=[],lb=[],ub=[],options={'disp': False},print_iter=False,**kwargs): # called when created
        
        self.model = model
        self.mom_data = mom_data
        self.mom_fun = mom_fun
        self.name = name

        # estimation settings
        self.options = options
        self.print_iter = print_iter
        self.method = method

        self.lb = lb
        self.ub = ub

        self.est_par = est_par


    def obj_fun(self,theta,W,*args):
        
        if self.print_iter:
            for p in range(len(theta)):
                print(f' {self.est_par[p]}={theta[p]:2.3f}', end='')

        # 1. update parameters 
        for i in range(len(self.est_par)):
            setattr(self.model.par,self.est_par[i],theta[i]) # like par.key = val

        # 2. solve model with current parameters
        self.model.solve()

        # 3. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
        self.model.simulate()
        self.mom_sim = self.mom_fun(self.model.sim,*args)

        # 4. calculate objective function and return it
        diff = self.mom_data - self.mom_sim
        self.obj  = (np.transpose(diff) @ W) @ diff

        if self.print_iter:
            print(f' -> {self.obj:2.4f}')

        return self.obj 

    def estimate(self,theta0,W,*args):
        # TODO: consider multistart-loop with several algortihms - that could alternatively be hard-coded outside
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        # estimate
        self.est_out = minimize(self.obj_fun, theta0, (W, *args), method=self.method,options=self.options)

        # return output
        self.est = self.est_out.x
        self.W = W

    def std_error(self,theta,W,Omega,Nobs,Nsim,step=1.0e-4,*args):
        ''' Calculate standard errors and sensitivity measures '''

        num_par = len(theta)
        num_mom = len(W[0])

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





# # moment function
# from numba import njit

# @njit
# def mom_fun(data):
#     low_noChild = np.mean(data.probs[:20, (data.states == 0) | (data.states == 4)], axis=1)*100
#     low_Child = np.mean(data.probs[:20, (data.states == 1) | (data.states == 5)], axis=1)*100
#     high_noChild = np.mean(data.probs[:20, (data.states == 2) | (data.states == 6)], axis=1)*100
#     high_Child = np.mean(data.probs[:20, (data.states == 3) | (data.states == 7)], axis=1)*100
#     return np.hstack((low_noChild, low_Child, high_noChild, high_Child))
        
# # create data
# import itertools
# from Model import RetirementModelClass
# states = list(itertools.product([0,1],repeat=4))
# states = states[:8]
# model = RetirementModelClass(states=states, a_max = 10, simN = 5000)
# model.solve()

# # Allocate states
# #model.par.simStates = funs.create_states(model,'female',0.5,0.5,0.5)
# ind = int(model.par.simN/8)
# states = np.hstack((0*np.ones(ind), 1*np.ones(ind), 2*np.ones(ind), 3*np.ones(ind), 
#                     4*np.ones(ind), 5*np.ones(ind), 6*np.ones(ind), 7*np.ones(ind)))
# states = np.array(states, dtype=int)
# model.par.simStates = states
# model.simulate()
# mom_data = mom_fun(model.sim)

# # prep
# weight = np.eye(len(mom_data))
# true = [model.par.alpha_0_female, model.par.alpha_1, model.par.alpha_2]
# theta0 = [i*3 for i in true]
# add_str = '_est'
# est_par = ("alpha_0_female", "alpha_1", "alpha_2") # remember to be list if only 1 var

# # Estimate the baseline model
# model_base = model
# model_base.prep = False

# smd_base = SimulatedMinimumDistance(model_base,mom_data,mom_fun,print_iter=True,options={'disp':True,'maxiter':10})
# smd_base.est_par = est_par
# smd_base.estimate(theta0,weight)