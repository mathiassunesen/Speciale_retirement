# global modules
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# consav
from consav import linear_interp

# local modules
import transitions
import funs

def policy(model,ax,var,T,AD,MA,ST,RA,D,label=False,xlim=None,ylim=None):
    """ plot either consumption or value functions
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        var (str): either 'c' for consumption or 'v' for value function
        time (list): list of times
        age_dif (list): list of age differences
        male (list): list of male indicator
        states (list): list of states
        choice (list): list of choices
        label (bool): if True (default) show labels
        xlim (list, optional): set x axis
        ylim (list, optional): set y axis

    Returns:
        ax (axes): axes object for plotting
    """

    # unpack
    sol = model.sol
    par = model.par
    solvardict = dict([('c','C_t'),
                       ('v','v_t')])    
    m = sol.m
    
    # loop through options
    for t in T:
        for ad in AD:
            for ma in MA:
                for st in ST:
                    for ra in RA:
                        for d in D:

                            if d == 1:
                                ra = transitions.ra_look_up(t,st,ra,d,par)
                            x = m[t,ad,ma,st,ra,d]
                            y = getattr(sol,var)[t,ad,ma,st,ra,d]
                            
                            if label == False:
                                ax.plot(x,y)
                            else:
                                if 't' in label and 'ra' in label and len(label) == 2:
                                    lab = f"$(t = {transitions.age(t,par)}, ra = {ra})$"
                                if 't' in label and len(label) == 1:
                                    lab = f"$(t = {transitions.age(t,par)})$"
                                ax.plot(x,y,label=lab)

    # details
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    if label:
        ax.legend()
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('${}$'.format(solvardict[var]))


def choice_probs(model,ax,AD,MA,ST,ages=[57,68],xlim=None,ylim=None):
    """ plot the average choice probabilities for multiple states (written to a common grid)
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        ages (list): list of ages        
        age_dif (int): age difference
        male (int): male indicator
        states (list): list of states
        xlim (list, optional): set x axis
        ylim (list, optional): set y axis

    Returns:
        ax (axes): axes object for plotting
    """

    # unpack
    sol = model.sol
    par = model.par
    v = sol.v

    # initalize
    ages = np.arange(ages[0], ages[1]+1)
    dims = (len(ST), len(ages))
    probs = np.empty(dims)
    
    # loop through options
    for st in ST:
        for t in transitions.inv_age(np.array(ages),par):
                    
            # average choice probabilities
            ra = transitions.ra_look_up(t,st,0,1,par)
            probs[st,t] = np.mean(funs.logsum2(v[t,AD,MA,st,ra],par)[1], axis=1)[0]

        # plot
        if transitions.state_translate(st,'elig',par)==1:
            lab = 'erp=1'
        else:
            lab = 'erp=0'
        if transitions.state_translate(st,'high_skilled',par)==1:
            lab = lab + ', hs=1'
        else:
            lab = lab + ', hs=0'
        ax.plot(ages,probs[st],'-o',label=lab)

    # details
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('age')
    ax.set_ylabel('Retirement probability')


def lifecycle(model,ax,vars=['m','c','a'],male=0,states=[0,1,2,3],ages=[57,68],quantiles=True):
    """ plot lifecycle
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        vars (list): con contain m, c, a, d and alive
        male (list): list of male indicator
        states (list): list of states
        ages (list): list with start age and end age
        quantiles (bool): if True (default) and only one element in vars then also plots lower and upper quartile

    Returns:
        ax (axes): axes object for plotting
    """

    # unpack
    sim = model.sim
    par = model.par

    # mask
    mask = np.nonzero(np.isin(sim.states,states))[0]
    mask = mask[sim.male[mask]==male]

    # figure
    simvardict = dict([('m','$m_t$'),
                  ('c','$C_t$'),
                  ('a','$a_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    x = np.arange(ages[0], ages[1]+1)
    for i in vars:
        simdata = getattr(sim,i)[transitions.inv_age(x,par)]
        y = simdata[:,mask]
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ax.plot(x,np.nanmean(y,axis=1),lw=2,label=simvardict[i])

            if len(vars)==1 and quantiles:
                ax.plot(x,np.nanpercentile(y,25,axis=1),'--',lw=1.5,label='lower quartile')
                ax.plot(x,np.nanpercentile(y,75,axis=1),'--',lw=1.5,label='upper quartile')
    
    # details
    ax.legend()
    ax.grid(True)    
    ax.set_xlabel('Age')
    if (len(x) < 15):
        ax.set_xticks(x)
    if ('m' in vars or 'c' in vars or 'a' in vars):
        ax.set_ylabel('100.000 DKR')


def retirement_probs(model,ax,male=0,states=[0,1,2,3],ages=[57,68],plot=True):
    """ plot retirement probabilities
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        male (list): list of male indicator
        states (list): list of states
        ages (list): list with start age and end age        

    Returns:
        ax (axes): axes object for plotting
    """    

    # unpack
    sim = model.sim
    par = model.par

    # mask
    mask = np.nonzero(np.isin(sim.states,states))[0]
    mask = mask[sim.male[mask]==male]    
    
    # figure
    x = np.arange(ages[0], ages[1]+1)
    y = sim.probs[transitions.inv_age(x,par)]   # ages
    y = y[:,mask]                               # states

    if plot:
        ax.plot(x,np.nanmean(y,axis=1),'r')
        
        # details
        ax.grid(True)    
        ax.set_xticks(x)
        ax.set_ylim(top=0.35)
        ax.set_xlabel('Age')
        ax.set_ylabel('Retirement probability')
    else:
        return x,np.nanmean(y,axis=1)
        

    
