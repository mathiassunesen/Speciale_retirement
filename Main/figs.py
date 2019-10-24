# global modules
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd

# consav
from consav import linear_interp

# local modules
import transitions
import funs

lw = 3
fs = 17

def plot_exp(x,Y,ax,labels,xlab,ylab):

    line_spec = ('-','-','--')
    colors = ('black', 'red', 'blue')

    for i in range(len(Y)):
        ax.plot(x,Y[i], label=labels[i], linewidth=lw, linestyle=line_spec[i], color=colors[i], marker='o')
    
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)    
    ax.legend(fontsize=fs-5)
    ax.tick_params(axis='both', which='major', labelsize=fs)

    return ax


def policy(model,ax,var,T,MA,ST,RA,D,label=False,xlim=None,ylim=None,bottom=0,top=False):
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
    if not top:
        top = len(par.grid_a)
    
    # loop through options
    ad = 0
    for t in T:
        for ma in MA:
            for st in ST:
                for ra in RA:
                    for d in D:

                        if d == 1:
                            ra = transitions.ra_look_up(t,st,ra,d,par)
                        x = m[t,ad,ma,st,ra,d,bottom:top]
                        y = getattr(sol,var)[t,ad,ma,st,ra,d,bottom:top]
                            
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

def policy_c(model,ax,var,T,AD,ST_h,ST_w,RA_h,RA_w,D_h,D_w,label=False,xlim=None,ylim=None,bottom=0):
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
                                    x = m[t,ad,st_h,st_w,ra_xh,ra_xw,d,bottom:]
                                    y = getattr(sol,var)[t,ad,st_h,st_w,ra_xh,ra_xw,d,bottom:]
                                    
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

def choice_probs(model,ax,AD,MA,ST,ages=[57,68],xlim=None,ylim=None):
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

    # initalize
    ages = np.arange(ages[0], ages[1]+1)
    dims = (len(ST), len(ages))
    probs = np.empty(dims)
    
    # loop through options
    for j in range(len(ST)):
        st = ST[j]
        for t in transitions.inv_age(np.array(ages),par):
                    
            # average choice probabilities
            ra = transitions.ra_look_up(t,st,0,1,par)
            probs[j,t] = np.mean(funs.logsum2(v[t,AD,MA,st,ra],par)[1], axis=1)[0]

        # plot
        if transitions.state_translate(st,'elig',par)==1:
            lab = 'erp=1'
        else:
            lab = 'erp=0'
        if transitions.state_translate(st,'high_skilled',par)==1:
            lab = lab + ', hs=1'
        else:
            lab = lab + ', hs=0'
        ax.plot(ages,probs[j],'-o',label=lab)

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
    ad_idx = ad + par.ad_min    # for look up in solution

    # initalize
    ages = np.arange(ages[0], ages[1]+1)
    dims = (len(ST), len(ages))
    probs = np.nan*np.zeros(dims)
    
    # loop through options
    for j in range(len(ST)):
        st_h = ST[j]
        st_w = ST[j]

        for time in transitions.inv_age(np.array(ages),par):
            if ma == 0:
                t = time+abs(ad)*(ad<0) # if wife is younger then rescale time, so we don't get fx 53,54,55 etc.
            elif ma == 1:
                t = time

            ra_h = transitions.ra_look_up(t,st_h,0,1,par)
            ra_w = transitions.ra_look_up(t+ad,st_w,0,1,par)
            prob = funs.logsum4(v[t,ad_idx,st_h,st_w,ra_h,ra_w], par)[1]
            if ma == 0:
                probs[j,time] = np.mean(prob[0]+prob[2])
            elif ma == 1:
                probs[j,time] = np.mean(prob[0]+prob[1])

        # plot
        if transitions.state_translate(ST[j],'elig',par)==1:
            lab = 'erp=1'
        else:
            lab = 'erp=0'
        if transitions.state_translate(ST[j],'high_skilled',par)==1:
            lab = lab + ', hs=1'
        else:
            lab = lab + ', hs=0'

        if ma == 0 and ad > 0:
            ax.plot(ages+ad,probs[j],'-o',label=lab)    # if wife is older, then her age start at 57+ad           
        else:
            ax.plot(ages,probs[j],'-o',label=lab)

    # details
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('age')
    ax.set_ylabel('Retirement probability')


def lifecycle(model,ax,vars=['m','c','a'],ma=0,ST=[0,1,2,3],ages=[57,68],quantiles=False,dots=False):
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
    states = sim.states
    MA = states[:,0]
    States = states[:,1]
    mask = np.nonzero(np.isin(States,ST))[0]
    mask = mask[MA[mask]==ma]

    # figure
    simvardict = dict([('m','$m_t$'),
                  ('c','$C_t$'),
                  ('a','$a_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    x = np.arange(ages[0], ages[1]+1)
    for i in vars:
        simdata = getattr(sim,i)[:,transitions.inv_age(x,par)]
        y = simdata[mask,:]
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if dots:
                ax.plot(x,np.nanmean(y,axis=0),'ko',lw=2,label=simvardict[i] + ' (Data)')
            else:
                ax.plot(x,np.nanmean(y,axis=0), 'r',lw=2,label=simvardict[i] + ' (Predicted)')                

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


def lifecycle_c(model,ax,vars=['m','c','a'],AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,68],quantiles=False,dots=False):
    """ plot lifecycle for couples"""

    # unpack
    sim = model.sim
    par = model.par

    # indices
    ADx = sim.states[:,0]
    STx_h = sim.states[:,1]
    STx_w = sim.states[:,2]
    idx = np.nonzero((np.isin(ADx,AD)) & ((np.isin(STx_h,ST_h)) & ((np.isin(STx_w,ST_w)))))[0]

    # figure
    simvardict = dict([('m','$m_t$'),
                  ('c','$C_t$'),
                  ('a','$a_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    x = np.arange(ages[0], ages[1]+1)
    for i in vars:
        simdata = getattr(sim,i)[:,transitions.inv_age(x,par)]
        y = simdata[idx,:]
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if dots:
                ax.plot(x,np.nanmean(y,axis=0),'ko',lw=2,label=simvardict[i] + ' (Data)')
            else:
                ax.plot(x,np.nanmean(y,axis=0), 'r',lw=2,label=simvardict[i] + ' (Predicted)')                

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

def retirement_probs(model,ax,MA=[0],ST=[0,1,2,3],ages=[58,68],moments=False):
    """ plot retirement probabilities for singles"""

    # unpack
    sim = model.sim
    par = model.par

    # indices
    MAx = sim.states[:,0]
    STx = sim.states[:,1]
    idx = np.nonzero((np.isin(MAx,MA) & np.isin(STx,ST)))[0]
    
    # figure
    x = np.arange(ages[0], ages[1]+1)
    y = sim.probs[:,transitions.inv_age(x,par)] # ages
    y = y[idx,:]                                # states

    # plot
    ax.plot(x,np.nanmean(y,axis=0), 'r', linewidth=lw, label='Predicted')
    if moments:
        mom_data = pd.read_excel('SASdata/single_moments_total.xlsx')/100        
        mom = np.reshape(mom_data['Moments'].to_numpy(), newshape=(int(len(mom_data)/3),3), order='F')
        if len(MA)>1:   # plot all
            ax.plot(x,mom[:,-1], 'ko', linewidth=lw, label='Data')
        else:   # plot either men or women
            ax.plot(x,mom[:,MA[0]], 'ko', linewidth=lw, label='Data')
        ax.legend(fontsize=fs)

    # details
    ax.grid(True)    
    ax.set_xticks(x)
    ax.tick_params(axis='both', which='major', labelsize=fs)    
    ax.set_ylim(top=0.35)
    ax.set_xlabel('Age', fontsize=fs)
    ax.set_ylabel('Retirement probability', fontsize=fs)
        

def retirement_probs_c(model,ax,ma=0,AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,68],plot=True,dots=False):
    """ plot retirement probabilities from couple model"""

    # unpack
    sim = model.sim
    par = model.par

    # indices
    ADx = sim.states[:,0]
    STx_h = sim.states[:,1]
    STx_w = sim.states[:,2]
    idx = np.nonzero((np.isin(ADx,AD) & (np.isin(STx_h,ST_h) & (np.isin(STx_w,ST_w)))))[0]
    
    # figure
    x = np.arange(ages[0], ages[1]+1)
    y = sim.probs[:,transitions.inv_age(x,par)+par.ad_min,ma]  # ages
    y = y[idx,:]                                               # states

    if plot:
        if dots:
            ax.plot(x,np.nanmean(y,axis=0),'ko',label='Data')
        else:
            ax.plot(x,np.nanmean(y,axis=0),'r',label='Predicted')
        
        # details
        ax.grid(True)    
        ax.set_xticks(x)
        ax.set_ylim(top=0.35)
        ax.set_xlabel('Age')
        ax.set_ylabel('Retirement probability')
    else:
        return x,np.nanmean(y,axis=1)        

def policy_simulation(model,var='d',MA=[0,1],ST=[0,1,2,3],ages=[57,110]):
    """ policy simulation for singles"""

    # unpack
    sim = model.sim
    par = model.par

    # indices
    MAx = sim.states[:,0]
    STx = sim.states[:,1]
    idx = np.nonzero(((np.isin(MAx,MA) & (np.isin(STx,ST)))))[0]
    x = np.arange(ages[0], ages[1]+1)
    
    # policy
    y = getattr(sim,var)[:,transitions.inv_age(x,par)]  # ages
    y = y[idx,:]                                        # states
    return np.nansum(y)

def policy_simulation_c(model,var='d',MA=[0,1],
                        AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,110]):
    """ policy simulation for couples"""

    # unpack
    sim = model.sim
    par = model.par

    # indices
    ADx = sim.states[:,0]
    STx_h = sim.states[:,1]
    STx_w = sim.states[:,2]
    idx = np.nonzero((np.isin(ADx,AD) & (np.isin(STx_h,ST_h) & (np.isin(STx_w,ST_w)))))[0]
    x = np.arange(ages[0], ages[1]+1)
    
    # policy
    y = getattr(sim,var)[:,transitions.inv_age(x,par)]  # ages
    y = y[idx,:]                                        # states
    return np.nansum(y)

def resolve(model,vars,recompute=True,accuracy=False,MA=[0,1],ST=[0,1,2,3],ages=[57,110],**kwargs):
    
    # dict
    store = {}
    for var in vars:
        store[var] = []

    # resolve model
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    for var in vars:
        for v in range(len(values[0])):
            
            # set new parameters
            for k in range(len(keys)):
                setattr(model.par,str(keys[k]),values[k][v])
            
            # solve and simulate
            model.solve(recompute=recompute)
            model.simulate(recompute=recompute,accuracy=accuracy)
            y = policy_simulation(model,var=var,MA=MA,ST=ST,ages=ages)
            store[var].append(y)

    # return
    return store

def resolve_c(model,vars,recompute=True,accuracy=False,
              MA=[0,1],AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,110],
              **kwargs):
    
    # dict
    store = {}
    for var in vars:
        store[var] = []

    # resolve model
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    for var in vars:
        for v in range(len(values[0])):
            
            # set new parameters
            for k in range(len(keys)):
                setattr(model.Single.par,str(keys[k]),values[k][v])
                setattr(model.par,str(keys[k]),values[k][v])
            
            # solve and simulate
            model.solve(recompute=recompute)
            model.simulate(recompute=recompute,accuracy=accuracy)
            y = policy_simulation_c(model,var=var,MA=MA,AD=AD,ST_h=ST_h,ST_w=ST_w,ages=ages)
            store[var].append(y)        

    # return
    return store

def plot_3D(dict,x_ax,y_ax,z_ax):
    
       
    xgrid = np.array(dict[x_ax])
    zgrid = np.array(dict[z_ax])
    ygrid = np.array(dict[y_ax])
    x, z = np.meshgrid(xgrid, zgrid)
    y, x = np.meshgrid(ygrid, xgrid)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,
                    y,
                    z,
                    rstride=2, cstride=2,
                    cmap=cm.jet,
                    alpha=0.7,
                    linewidth=0.25)
    ax.set_xlabel(x_ax)
    ax.set_ylabel(y_ax)
    ax.set_zlabel(z_ax)
    ax.set_ylim([0,100])
    plt.show()    

def plot_3DD(dict,x_ax,y_ax,z_ax):
    #Sort data:
    x = dict[x_ax]
    y = dict[y_ax]
    z = dict[z_ax]

    idx = np.argsort(x)
    x_sort = []
    y_sort = []
    z_sort = []
    for i in idx:
        x_sort.append(x[i])
        y_sort.append(y[i])
        z_sort.append(z[i])
    #Plot data
    xgrid = np.array(x_sort)
    zgrid = np.array(z_sort)
    ygrid = np.array(y_sort)
    x, z = np.meshgrid(xgrid, zgrid)
    y, x = np.meshgrid(ygrid, xgrid)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x,
                    y,
                    z,
                    rstride=2, cstride=2,
                    cmap=cm.jet,
                    alpha=0.7,
                    linewidth=0.25)
    ax.set_xlabel(x_ax)
    ax.set_ylabel(y_ax)
    ax.set_zlabel(z_ax)
    plt.show()   

def plot_2DD(dict,x_ax,y_ax,xlim,ylim):
    #Sort data
    x = dict[x_ax]
    y = dict[y_ax]
    idx = np.argsort(x)

    x_sort = []
    y_sort = []
    for i in idx:
        x_sort.append(x[i])
        y_sort.append(y[i])
    
    #Plot data:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.plot(x_sort,y_sort)
    # c. details
    ax.set_ylim([ylim[0],ylim[1]])
    ax.set_xlim([xlim[0],xlim[1]])
    ax.set(xlabel=x_ax, ylabel=y_ax)