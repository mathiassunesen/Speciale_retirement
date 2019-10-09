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

    def __init__(self,model,mom_data,mom_fun,name='baseline',method='nelder-mead',est_par=[],est_par_save={},lb=int,ub=int,guess=[],options={'disp': False},print_iter=[False,1],save=False,**kwargs): # called when created
        
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

        self.save = save #se efter save og est_par_save - når dette skal ligges over i main.
        self.est_par = est_par
        self.est_par_save = est_par_save
        self.iter = 0


    def obj_fun(self,theta,W,*args):
        
        self.iter += 1

        if self.print_iter[0]:
            if self.iter % self.print_iter[1] == 0:
                print('Iteration: ', self.iter)
                for p in range(len(theta)):
                    print(f' {self.est_par[p]}={theta[p]:2.3f}', end='')

        # 1. update parameters 
        for i in range(len(self.est_par)):
            setattr(self.model.par,self.est_par[i],theta[i]) # like par.key = val

        # 2. solve model with current parameters
        self.model.solve()

        # 3. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
        self.model.simulate()
        self.mom_sim = self.mom_fun(self.model,*args)

        # 4. calculate objective function and return it
        diff = self.mom_data - self.mom_sim
        self.obj  = (np.transpose(diff) @ W) @ diff

        if self.print_iter[0]:
            if self.iter % self.print_iter[1] == 0:
                print(f' -> {self.obj:2.4f}')
        
        if self.save:
            for p in range(len(theta)):
                self.est_par_save[self.est_par[p]].append(theta[p])
            self.est_par_save['obj_func'].append(self.obj)

        return self.obj 

    def estimate(self,theta0,W,*args):
        # TODO: consider multistart-loop with several algortihms - that could alternatively be hard-coded outside
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        # estimate
        self.est_out = minimize(self.obj_fun, theta0, (W, *args), method=self.method,options=self.options)

        # return output
        self.est = self.est_out.x
        self.W = W     

         
    def multistart_estimate(self,guess,W,*args):
        # TODO: consider multistart-loop with several algortihms - that could alternatively be hard-coded outside
        
  
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data
        
        for i in guess:
            # estimate
            self.est_out = minimize(self.obj_fun, i, (W, *args), method=self.method,options=self.options)

        # return output
        self.est = self.est_out.x
        self.W = W
    
    def multistart_V(self, ng, guess):
        # ng = number of guess - number of starting points
        # guess: list of variables and the lower and upper bound of starting points.
    
        #Laver en dict med start værdier for hver parameter vi vil estimer:
        q = {'{}'.format(i):[] for i in self.est_par}
        for key in guess:
            for i in range(ng):
                q[key].append(np.random.uniform(guess[key][0],guess[key][1]))
    
        #Laver en liste med hvert gæt - dvs en startværdi for hver parameter - denne bruges som insdput i estimate.
        start_values = [[] for i in range(ng)]
        for i in range(ng):
            for key in q:
                start_values[i].append(q[key][i])
        return start_values

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

