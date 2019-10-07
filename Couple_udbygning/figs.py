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

def policy(model,ax,var,T,MA,ST,RA,D,label=False,xlim=None,ylim=None):
    """ plot either consumption or value functions for the single model
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        var (str): either 'c' for consumption or 'v' for value function
        T (list): list of times
        MA (list): list of male indicator
        ST (list): list of states
        D (list): list of choices
        label (list, optional): list of what to show in label
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
    ad = 0
    for t in T:
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

def policy_c(model,ax,var,T,AD,ST_h,ST_w,RA_h,RA_w,D_h,D_w,label=False,xlim=None,ylim=None):
    """ plot either consumption or value functions
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        var (str): either 'c' for consumption or 'v' for value function
        T (list): list of times
        AD (list): list of age differences
        ST_h (list): list of states, husband
        ST_h (list): list of states, wife
        D_h (list): list of choices, husband
        D_w (list): list of choices, wife        
        label (list, optional): list of what to show in label
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
    ad_min = par.ad_min
    
    # loop through options
    for t in T:
        for ad in AD:
            ad = ad + ad_min
            for st_h in ST_h:
                for st_w in ST_w:
                    for ra_h in RA_h:
                        for ra_w in RA_w:
                            for d_h in D_h:
                                for d_w in D_w:

                                    if d_h == 1:
                                        ra_xh = transitions.ra_look_up(t,st_h,ra_h,d_h,par)
                                    else:
                                        ra_xh = ra_h
                                    if d_w == 1:
                                        ra_xw = transitions.ra_look_up(t,st_w,ra_w,d_w,par)
                                    else:
                                        ra_xw = ra_w

                                    d = transitions.d_c(d_h,d_w)
                                    x = m[t,ad,st_h,st_w,ra_xh,ra_xw,d]
                                    y = getattr(sol,var)[t,ad,st_h,st_w,ra_xh,ra_xw,d]
                                    
                                    if label == False:
                                        ax.plot(x,y)
                                    else:
                                        # if 't' in label and 'ra' in label and len(label) == 2:
                                        #     lab = f"$(t = {transitions.age(t,par)}, ra = {ra})$"
                                        if 't' in label and len(label) == 1:
                                            lab = f"$(t = {transitions.age(t,par)})$"
                                        elif 'd' in label and len(label) == 1:
                                            lab = f"$(d^h = {d_h}, d^w = {d_w})$"
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

def choice_probs(model,ax,MA,ST,ages=[57,68],xlim=None,ylim=None):
    """ plot the average choice probabilities for multiple states
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        ages (list): list with start age and end age        
        MA (int): male indicator
        ST (list): list of states
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
            probs[st,t] = np.mean(funs.logsum2(v[t,0,MA,st,ra],par)[1], axis=1)[0]

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


def choice_probs_c(model,ax,ma,ST,ad=0,ages=[57,68],xlim=None,ylim=None):
    """ plot the average choice probabilities for multiple states
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        ages (list): list with start age and end age        
        AD (int): age difference
        MA (int): male indicator
        ST (list): list of states
        xlim (list, optional): set x axis
        ylim (list, optional): set y axis

    Returns:
        ax (axes): axes object for plotting
    """

    # unpack
    sol = model.sol
    par = model.par
    v = sol.v
    ad_min = par.ad_min

    # initalize
    ages = np.arange(ages[0], ages[1]+1)
    if ma == 0:
        ages = ages + ad
    dims = (len(ST), len(ages))
    probs = np.empty(dims)
    
    # loop through options
    for st in ST:
        for t in transitions.inv_age(np.array(ages),par):
                    
            # average choice probabilities
            ra = transitions.ra_look_up(t,st,0,1,par)
            if ma == 0:
                prob = funs.logsum4(v[t,ad+ad_min,0,st,2,ra], par)[1]
                prob = prob[0] + prob[2]
                probs[st,t] = np.mean(prob)
            elif ma == 1:
                prob = funs.logsum4(v[t,ad+ad_min,st,0,ra,2], par)[1]
                prob = prob[0] + prob[1]
                probs[st,t] = np.mean(prob)

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


def lifecycle(model,ax,vars=['m','c','a'],ma=0,ST=[0,1,2,3],ages=[57,68],quantiles=True):
    """ plot lifecycle
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        vars (list): con contain m, c, a, d and alive
        ma (int): male indicator
        ST (list): list of states
        ages (list): list with start age and end age
        quantiles (bool): if True (default) and only one element in vars then also plots lower and upper quartile

    Returns:
        ax (axes): axes object for plotting
    """

    # unpack
    sim = model.sim
    par = model.par

    # mask
    mask = np.nonzero(np.isin(sim.ST,ST))[0]
    mask = mask[sim.MA[mask]==ma]

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


def retirement_probs(model,ax,ma=0,ST=[0,1,2,3],ages=[57,68],plot=True):
    """ plot retirement probabilities
    
    Args:
        model (class): parameters, solution and simulation
        ax (axes): axes object for plotting
        ma (int): male indicator
        ST (list): list of states
        ages (list): list with start age and end age        

    Returns:
        ax (axes): axes object for plotting
    """    

    # unpack
    sim = model.sim
    par = model.par

    # mask
    mask = np.nonzero(np.isin(sim.ST,ST))[0]
    mask = mask[sim.MA[mask]==ma]    
    
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
        

    
