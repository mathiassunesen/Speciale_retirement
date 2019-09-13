import numpy as np
from numba import njit, prange
import time
from prettytable import PrettyTable
from consav import linear_interp
import transitions

def GaussHermite(n):
    return np.polynomial.hermite.hermgauss(n)

def GaussHermite_lognorm(sigma, n):
    x,w = GaussHermite(n)

    x = np.exp(x*np.sqrt(2)*sigma - 0.5*sigma**2)
    w = w/np.sqrt(np.pi)

    assert(1 - sum(w*x) < 1e-8)
    return x,w    


@njit(parallel=True)
def logsum_vec(V, par):
    
    # 1. setup
    sigma = par.sigma_eta
    # if len(V.shape) == 1: # to be compatible with simulate which is loop based
    #     V = V.reshape(1,len(V))

    cols = V.shape[1]

    # 2. maximum over the discrete choices
    #if cols == 2: # 2 choices for singles
    mxm = np.maximum(V[:,0],V[:,1]).reshape(len(V),1)
    # elif cols == 4: # 4 choices for couples
    #     mxm = np.maximum(V[:,0],V[:,1],V[:,2],V[:,3]).reshape(len(V),1)        

    # 3. logsum and probabilities
    if abs(sigma) > 1e-10:
        logsum = mxm + sigma*np.log(np.sum(np.exp((V - mxm*np.ones((1,cols)))/sigma), 
                                            axis=1)).reshape(mxm.shape)
        prob = np.exp((V - logsum*np.ones((1,cols)))/sigma)

    else:
        logsum = mxm
        prob = np.zeros(V.shape)
        for i in range(len(V)): # only works for 2 choices
            if V[i,0] > V[i,1]:
                prob[i,0] = 1
            else:
                prob[i,1] = 1 

    return logsum,prob


def random_draws(sim,par):

    # a. set seed
    np.random.seed(par.sim_seed)

    # b. random draws
    sim.unif = np.random.rand(par.simT,par.simN)            # taste shocks
    sim.deadP = np.random.rand(par.simT,par.simN)           # death probs
    
    # c. income shocks
    sim.inc_shock = np.nan*np.zeros((par.Tr-1,par.simN))    # initalize
    num_st = np.unique(sim.states)
    for st in num_st:
        mask = np.nonzero(sim.states==st)[0]
        if transitions.state_translate(st,'male',par) == 1:
            sim.inc_shock[:,mask] = np.random.lognormal(-0.5*(par.sigma_xi_men**2),par.sigma_xi_men,size=(par.Tr-1,mask.size))

        else:
            sim.inc_shock[:,mask] = np.random.lognormal(-0.5*(par.sigma_xi_women**2),par.sigma_xi_women,size=(par.Tr-1,mask.size))








def my_timer(funcs,argu,names,Ntimes=100,unit='ms',ndigits=2,numba_disable=0,numba_threads=8):
    '''wrapper'''

    from consav import runtools
    runtools.write_numba_config(disable=numba_disable,threads=numba_threads)

    # check if correct type
    if type(funcs) == list:
        pass
    else:
        funcs = [funcs]
    
    # allocate
    store = np.empty((Ntimes,len(funcs)))
    def run_func(f,*args,**kwargs):
        f(*args,**kwargs)    
        
    for i in range(len(funcs)): # loop over funcions
        for j in range(Ntimes): # loop over executions
            tic = time.time()
            run_func(funcs[i],*argu[funcs[i]])
            toc = time.time()
            store[j,i] = toc-tic
            
    out,unit = timings(funcs,names,store,Ntimes,unit,ndigits)
    print('time unit is:',unit)
    print(out)
        
def timings(funcs, names, store, Ntimes, unit, ndigits):

    # prep
    header = ['func', 'lq', 'median', 'mean', 'uq', 'neval']
    
    # summary statistics
    summ = np.transpose(np.vstack(
                              (np.percentile(store, [25,50], axis=0), #lq, median
                               np.mean(store, axis=0),                      #mean
                               np.percentile(store, 75, axis=0))            #uq
                               ))              
    
    # format times
    factor = {'ms':1E3, 'us':1E6, 'ns':1E9, 's':1}
    summ = np.round(summ*factor[unit], ndigits)
    
    # add neval
    summ = np.hstack((summ, np.ones((len(funcs),1))*Ntimes))
    
    # create table
    out = PrettyTable(header)
    for i in range(len(funcs)):
        out.add_row([names[funcs[i]]]+(summ[i,:].tolist()))
    
    # output
    return out,unit


def create_states(model,sex,elig_frac,hs_frac):
    
    # solve for the optimal numbers
    x,add = states_sol(model,sex,elig_frac,hs_frac)
    
    # create array for states
    mm = states_array(x,sex,add)
    
    # check fractions
    print('fractions:', check_fracs(mm,model))
    return mm

def states_sol(model,sex,elig_frac,hs_frac):
    
    # imports
    import numpy as np
    from scipy.optimize import nnls 

    # set up equation
    # lhs
    if len(model.par.states) > 4:
        if sex == 'male':
            states = model.par.states[4:]
        else:
            states = model.par.states[:4]

        add = 4
    
    else:
        states = model.par.states
        add = 0
    
    if type(states) == list:
        A = np.transpose(np.array(states))
    else:
        A = np.transpose(states)
    A[0,:] = 1 # assure 1 col is ones

    # rhs
    elig = round(elig_frac*model.par.simN)
    hs = round(hs_frac*model.par.simN)
    b = np.array([model.par.simN, elig, hs])

    # solution
    x, rnorm = nnls(A,b)
    assert rnorm<=1e-6, 'residual is not zero' # assure residual is zero
    return x,add

def states_array(x, sex, add):
    
    lst = []
    x = np.around(x)
    x = x.astype(int)

    if sex == 'male':
        pass # gets add as argument
    elif sex == 'female':
        add = 0

    for i in range(len(x)):
        lst.append((i+add)*np.ones(x[i]))

    mm = np.concatenate(lst)
    mm = mm.astype(int)
    return mm

def check_fracs(mm,model):
    
    import transitions

    fracs = np.zeros((len(mm),2))

    for i in range(len(mm)):
        fracs[i,0] = transitions.state_translate(mm[i],'elig',model.par)
        fracs[i,1] = transitions.state_translate(mm[i],'high_skilled',model.par)
    
    return np.mean(fracs, axis=0)