# global modules
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
plt.style.use('ggplot')

# consav
from consav import linear_interp

# local modules
import transitions
import funs
lw = 3
fs = 17
def MyPlot(G,xlim=None,ylim=None,save=True,**kwargs):
    """ wrapper for plotting """
    
    # initialize
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    def update(g):
        fs = 17
        if not 'fontsize' in g:
            g['fontsize'] = fs
        if not 'labelsize' in g:
            g['labelsize'] = fs
        if not 'legendsize' in g:
            g['legendsize'] = fs
        if not 'linestyle' in g:
            g['linestyle'] = ('-',)*len(g['y'])
        if not 'marker' in g:
            g['marker'] = None

    def graph(g):
        for i in range(len(g['y'])):
            if 'color' in g:
                ax.plot(g['x'],g['y'][i],label=g['label'][i],color=g['color'][i],linestyle=g['linestyle'][i],marker=g['marker'],**kwargs)            
            else:
                ax.plot(g['x'],g['y'][i],label=g['label'][i],linestyle=g['linestyle'][i],marker=g['marker'],**kwargs)        

    if type(G) == list:
        for i in range(len(G)):
            g = G[i]
            update(g)
            graph(g)
    else:
        g = G
        update(g)
        graph(g)

    # details
    ax.legend(fontsize=g['legendsize'])
    ax.set_xlabel(g['xlabel'], fontsize=g['fontsize'])
    ax.set_ylabel(g['ylabel'], fontsize=g['fontsize'])
    if 'xticks' in g:
        ax.set_xticks(g['xticks'])
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)    
    ax.tick_params(axis='both', which='major', labelsize=g['labelsize'])        
    fig.tight_layout()
    
    if save:
        return fig


def plot_exp(x,Y,ax,labels,xlab,ylab):
    """ plot counterfactual policy simulations/experiments
    
        Args:
            x (1d array)
            Y (list of 1d arrays)
    """

    # plot
    for i in range(len(Y)):
        ax.plot(x,Y[i], label=labels[i], linewidth=lw, linestyle=line_spec[i], color=colors[i], marker='o')    
    
    # details
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)    
    ax.legend(fontsize=fs-5)
    ax.tick_params(axis='both', which='major', labelsize=fs)

def policy(model,var,T,MA,ST,RA,D,label=False,xlim=None,ylim=None,bottom=0,top=False):

    # unpack
    sol = model.sol
    par = model.par
    solvardict = dict([('c','C_t'),
                       ('v','v_t')])    
    y_lst = []
    label_lst = []
    m = sol.m
    if not top:
        top = len(m)
    
    # loop through options
    for age in T:
        for ma in MA:
            for st in ST:
                for ra in RA:
                    for d in D:

                        t = transitions.inv_age(age,par)
                        if d == 1:
                            ra = transitions.ra_look_up(t,st,ra,d,par)
                        x = m[bottom:top]
                        y = getattr(sol,var)[t,ma,st,ra,d,bottom:top]
                        y_lst.append(y)      

                        if not label:
                            lab = None
                        else:
                            if 't' in label and 'ra' in label and len(label) == 2:
                                lab = f"$(t = {age}, ra = {ra})$"
                            if 't' in label and len(label) == 1:
                                lab = f"$(t = {age})$"
                        label_lst.append(lab)
    # return
    return {'y': y_lst, 'x': x, 'label': label_lst, 'xlabel': '$m_t$', 'ylabel': '${}$'.format(solvardict[var])}

def choice_probs(model,ma,ST=[0,1,2,3],ages=[57,67]):

    # unpack
    sol = model.sol
    par = model.par
    v = sol.v

    # initalize
    ages = np.arange(ages[0], ages[1]+1)
    dims = (len(ST), len(ages))
    probs = np.empty(dims)
    label = []
    
    # loop through options
    for j in range(len(ST)):
        st = ST[j]
        for t in transitions.inv_age(np.array(ages),par):
                    
            # average choice probabilities
            ra = transitions.ra_look_up(t,st,0,1,par)
            probs[j,t] = np.mean(funs.logsum2(v[t,ma,st,ra],par)[1], axis=1)[0]

        # labels
        if transitions.state_translate(st,'elig',par)==1:
            lab = 'erp=1'
        else:
            lab = 'erp=0'
        if transitions.state_translate(st,'high_skilled',par)==1:
            lab = lab + ', hs=1'
        else:
            lab = lab + ', hs=0'
        label.append(lab)

    # return (+1 to recenter timeline)
    return {'y': probs, 'x': ages+1, 'xticks': ages+1, 'xlabel': 'Age', 'ylabel': 'Retirement probability', 'label': label}

def choice_probs_c(model,ma,ST=[0,1,2,3],ad=0,ages=[57,67]):
    """ plot the average choice probabilities for couples across time and states for a given age difference.
        Assuming the same state for both wife and husband """

    # unpack
    sol = model.sol
    par = model.par
    v = sol.v
    ad_idx = ad+par.ad_min    # for look up in solution

    # initalize
    ages = np.arange(ages[0], ages[1]+1)
    dims = (len(ST), len(ages))
    probs = np.nan*np.zeros(dims)
    label = []    
    
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

        # labels
        if transitions.state_translate(ST[j],'elig',par)==1:
            lab = 'erp=1'
        else:
            lab = 'erp=0'
        if transitions.state_translate(ST[j],'high_skilled',par)==1:
            lab = lab + ', hs=1'
        else:
            lab = lab + ', hs=0'
        label.append(lab)        

    # return (+1 to recenter timeline)
    if ma == 0:
        x = ages+ad+1
    elif ma == 1:
        x = ages+1
    return {'y': probs, 'x': x, 'xticks': x, 'xlabel': 'Age', 'ylabel': 'Retirement probability', 'label': label}

def lifecycle(model,var,MA=[0],ST=[0,1,2,3],ages=[57,68]):
    """ plot lifecycle """

    # unpack
    sim = model.sim
    par = model.par

    # indices
    MAx = sim.states[:,0]
    STx = sim.states[:,1]
    idx = np.nonzero((np.isin(MAx,MA) & np.isin(STx,ST)))[0]
    
    # figure
    simvardict = dict([('m','$m_t$'),
                  ('c','$C_t$'),
                  ('a','$a_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    x = np.arange(ages[0], ages[1]+1)
    simdata = getattr(sim,var)[:,transitions.inv_age(x,par)]
    y = simdata[idx,:]
    with warnings.catch_warnings(): # ignore this specific warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y = np.nanmean(y,axis=0)

    # return
    if var in ('m', 'c', 'a'):
        ylabel = '100.000 DKR'
    if var in ('d', 'alive'):
        ylabel = 'Share'
    return {'y': [y], 'x': x, 'xlabel': 'Age', 'ylabel': ylabel, 'label': [simvardict[var]]}

def lifecycle_c(model,var,AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,68]):
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
    simdata = getattr(sim,var)[:,transitions.inv_age(x,par)]
    y = simdata[idx,:]
    with warnings.catch_warnings(): # ignore this specific warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y = np.nanmean(y,axis=0)
    
    # return
    if var in ('m', 'c', 'a'):
        ylabel = '100.000 DKR'
    if var in ('d', 'alive'):
        ylabel = 'Share'
    return {'y': [y], 'x': x, 'xlabel': 'Age', 'ylabel': ylabel, 'label': [simvardict[var]]}

def retirement_probs(model,MA=[0],ST=[0,1,2,3],ages=[58,68]):
    """ plot retirement probabilities for singles """

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
    y = np.nanmean(y,axis=0)

    # return
    return {'y': [y], 'x': x, 'xticks': x, 'xlabel': 'Age', 'ylabel': 'Retirement probability', 'label': ['Predicted']}

def retirement_probs_c(model,ma,AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[58,68]):
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
    y = np.nanmean(y,axis=0)    
    
    # return
    return {'y': [y], 'x': x, 'xticks': x, 'xlabel': 'Age', 'ylabel': 'Retirement probability', 'label': ['Predicted']}

def policy_c(model,ax,var,T,AD,ST_h,ST_w,RA_h,RA_w,D_h,D_w,label=False,xlim=None,ylim=None,bottom=0):
    """ plot either consumption or value functions for couples """

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
                                    x = m[bottom:]
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

def policy_simulation(model,var,MA=[0,1],ST=[0,1,2,3],ages=[57,110]):
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

def policy_simulation_c(model,var,AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,110]):
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

def resolve(model,vars,recompute=True,accuracy=False,tax=True,MA=[0,1],ST=[0,1,2,3],ages=[57,110],**kwargs):
    
    # dict
    store = {}
    for var in vars:
        store[var] = []

    # resolve model
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    for v in range(len(values[0])):
        
        # set new parameters
        for k in range(len(keys)):
            setattr(model.par,str(keys[k]),values[k][v])

        # solve and simulate
        model.solve(recompute=recompute)
        model.simulate(accuracy=accuracy,tax=tax)

        # policy
        for var in vars:
            y = policy_simulation(model,var=var,MA=MA,ST=ST,ages=ages)
            store[var].append(y)

    # return
    return store 

def resolve_c(model,vars,recompute=True,accuracy=False,tax=True,
              AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,110],**kwargs):

    # dict
    store = {}
    for var in vars:
        store[var] = []

    # resolve model
    keys = list(kwargs.keys())
    values = list(kwargs.values())
    for v in range(len(values[0])):
            
        # set new parameters
        for k in range(len(keys)):
            setattr(model.Single.par,str(keys[k]),values[k][v])
            setattr(model.par,str(keys[k]),values[k][v])
            
        # solve and simulate
        model.solve(recompute=recompute)
        model.simulate(accuracy=accuracy,tax=tax)

        # policy
        for var in vars:
            y = policy_simulation_c(model,var=var,AD=AD,ST_h=ST_h,ST_w=ST_w,ages=ages)
            store[var].append(y)

    # return
    return store

def sens_fig_tab(sens,sense,theta,est_par_tex,fixed_par_tex):
    
    fs = 17
    sns.set(rc={'text.usetex' : False})
    cmap = sns.diverging_palette(10, 220, sep=10, n=100)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax = sns.heatmap(sense,annot=True,fmt="2.2f",annot_kws={"size": fs},xticklabels=fixed_par_tex,yticklabels=est_par_tex,center=0,linewidth=.5,cmap=cmap)
    plt.yticks(rotation=0) 
    ax.tick_params(axis='both', which='major', labelsize=20)