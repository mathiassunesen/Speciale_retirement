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

def LaborSupply(LS,indi,ages,start_age):
    x = np.arange(ages[0], ages[1]+1)
    x_inv = x - start_age

    if indi == 'educ':
        hs = LS['d'][1]['hs'][x_inv]-LS['d'][0]['hs'][x_inv]
        ls = LS['d'][1]['ls'][x_inv]-LS['d'][0]['ls'][x_inv]
        assert np.allclose(hs+ls,LS['d'][1]['base'][x_inv]-LS['d'][0]['base'][x_inv])

    elif indi == 'educ_w':
        hs = LS['d'][1]['hs_f'][x_inv]-LS['d'][0]['hs_f'][x_inv]
        ls = LS['d'][1]['ls_f'][x_inv]-LS['d'][0]['ls_f'][x_inv]
        assert np.allclose(hs+ls,LS['d'][1]['base_f'][x_inv]-LS['d'][0]['base_f'][x_inv])

    elif indi == 'educ_m':
        hs = LS['d'][1]['hs_m'][x_inv]-LS['d'][0]['hs_m'][x_inv]
        ls = LS['d'][1]['ls_m'][x_inv]-LS['d'][0]['ls_m'][x_inv]
        assert np.allclose(hs+ls,LS['d'][1]['base_m'][x_inv]-LS['d'][0]['base_m'][x_inv])

    elif indi == 'gender':
        hs = LS['d'][1]['base_f'][x_inv]-LS['d'][0]['base_f'][x_inv]
        ls = LS['d'][1]['base_m'][x_inv]-LS['d'][0]['base_m'][x_inv]
        assert np.allclose(hs+ls,LS['d'][1]['base'][x_inv]-LS['d'][0]['base'][x_inv])        

    return hs,ls,x

def LS_bar(LS,indi,N,start_age,ages,fs=17,ls=12,save=True):
    
    # set up
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # compute
    HS,LS,x = LaborSupply(LS,indi,ages,start_age)
    
    # plot
    if indi == 'gender':
        label = ['Women', 'Men']
    else:
        label = ['High skilled', 'Low skilled']
    ax.bar(x,HS*100/N, label=label[0])
    ax.bar(x,LS*100/N,bottom=HS*100/N, label=label[1])
    ax.legend(fontsize=ls)
    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Pct. change in labor supply',fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=ls)  
    fig.tight_layout()     

    if save:
        return fig      

def RetAge_linear(G,x='base'):
    age = []
    for i in range(len(G)):
        age.append(G[i][x])

    age = np.array(age)
    return age[1:]-age[0]

def RetAge_plot(x,Y,labels,xlab,ylab,lw=3,fs=17,ls=12,line_45=True,save=True):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plot
    if line_45:
        ax.plot(x,x, linewidth=lw, label='45 degree line', linestyle='--', color='k')
    for i in range(len(Y)):
        g = RetAge_linear(Y[i]['RetAge'])
        ax.plot(x,g, linewidth=lw, label=labels[i], marker='o')    
    
    # details
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)    
    ax.legend(fontsize=ls)
    ax.tick_params(axis='both', which='major', labelsize=ls)
    fig.tight_layout()

    if save:
        return fig                

def Surplus(G,N,change='Pct'):
    g = np.array(G['GovS'])
    if change == 'Pct':
        return (g[1:]-g[0])/abs(g[0])*100
    elif change == 'Abs':
        return (g[1:]-g[0])/N

def GovS_plot(x,Y,labels,xlab,ylab,lw=3,fs=17,ls=12,N=[1,1,1],change='Pct',save=True):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plot
    for i in range(len(Y)):
        g = Surplus(Y[i],N[i],change)
        ax.plot(x,g, linewidth=lw, label=labels[i], marker='o')    
    
    # details
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)    
    ax.legend(fontsize=ls)
    ax.tick_params(axis='both', which='major', labelsize=ls)
    fig.tight_layout()

    if save:
        return fig

def GovS_pct_change_plot(x,Y,labels,xlab,ylab,lw=3,fs=17,ls=12,save=True):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    # pre
    lst = []
    for i in range(len(Y)):
        lst.append(Surplus(Y[i],1,'Abs'))
    g1 = (np.array(lst[0])/np.array(lst[1])-1)*100
    g2 = (np.array(lst[0])/np.array(lst[2])-1)*100    

    # plot
    lns1 = ax1.plot(x,g1, linewidth=lw, label=labels[0], marker='o') 
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x,g2, 'b', marker='o', linewidth=lw, label=labels[1])     
    
    # details
    ax1.set_xlabel(xlab, fontsize=fs)
    ax1.set_ylabel(ylab, fontsize=fs)    
    ax1.legend(fontsize=ls)
    ax1.tick_params(axis='both', which='major', labelsize=ls)
    ax2.set_xlabel(xlab, fontsize=fs)
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=ls)
    ax1.set_ylim([np.min(g1)*0.8, np.max(g1)*1.2])
    ax2.set_ylim([np.min(g2)*0.8, np.max(g2)*1.2])        
    ax2.tick_params(axis='both', which='major', labelsize=ls)
    ax2.grid(False)    
    fig.tight_layout()

    if save:
        return fig    

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

def lifecycle(model,var,MA=[0],ST=[0,1,2,3],ages=[57,68],calc='mean'):
    """ plot lifecycle """

    if var == 'probs':
        return retirement_probs(model,MA,ST,ages)
    else:

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
                    ('GovS', '$G_t'),
                    ('d','$d_t$'),
                    ('alive','$alive_t$')])

        x = np.arange(ages[0], ages[1]+1)
        simdata = getattr(sim,var)[:,transitions.inv_age(x,par)]
        y = simdata[idx,:]
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if calc == 'mean':
                y = np.nanmean(y,axis=0)
            elif calc == 'sum':
                y = np.nansum(y,axis=0)
            elif calc == 'total_sum':
                y = np.nansum(y)

        # return
        if var in ('m', 'c', 'a', 'GovS'):
            ylabel = '100.000 DKR'
        if var in ('d', 'alive'):
            ylabel = 'Share'
        return {'y': [y], 'x': x, 'xlabel': 'Age', 'ylabel': ylabel, 'label': [simvardict[var]]}

def lifecycle_c(model,var,MA=[0,1],AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[57,68],calc='mean'):
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
                  ('GovS','$G_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    x = np.arange(ages[0], ages[1]+1)
    axis = 0
    if var in ('m', 'c', 'a', 'GovS'):
        simdata = getattr(sim,var)[:,transitions.inv_age(x,par)]
    elif var in ('d', 'alive'):
        simdata = getattr(sim,var)[:,transitions.inv_age(x,par)+par.ad_min]
        simdata = simdata[:,:,MA]
        if len(MA)>1:
            axis = (0,2)
    
    y = simdata[idx]
    with warnings.catch_warnings(): # ignore this specific warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if calc == 'mean':
            y = np.nanmean(y,axis=axis)
            y = y.ravel()
        elif calc == 'sum':
            y = np.nansum(y,axis=axis)
            y = y.ravel()
        elif calc == 'total_sum':
            y = np.nansum(y)
    
    # return
    if var in ('m', 'c', 'a', 'GovS'):
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
    with warnings.catch_warnings(): # ignore this specific warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y = np.nanmean(y,axis=0)    
    
    # return
    return {'y': [y], 'x': x, 'xticks': x, 'xlabel': 'Age', 'ylabel': 'Retirement probability', 'label': ['Predicted']}

def RetAge_S(model,MA=[0,1],ST=[0,1,2,3]):
    
    par = model.par
    sim = model.sim
    
    # individuals not dying before retiring and selected group
    MAx = sim.states[:,0]
    STx = sim.states[:,1]
    idx = np.nonzero((np.any(sim.d==0,axis=1)) & (np.isin(MAx,MA)) & (np.isin(STx,ST)))[0]
    
    # distribution of retirement ages
    age = np.nanmax(np.arange(par.start_T,par.start_T+par.simT)*sim.d[idx],axis=1)+1

    # return
    return age

def RetAge_C(model,ma,AD=[-4,-3,-2,-1,0,1,2],ST_h=[0,1,2,3],ST_w=[0,1,2,3]):

    par = model.par
    sim = model.sim

    # individuals not dying before retiring and selected group
    ADx = sim.states[:,0]
    ST_hx = sim.states[:,1]
    ST_wx = sim.states[:,2]
    idx = np.nonzero((np.any(sim.d[:,:,ma]==0,axis=1)) & 
                     (np.isin(ADx,AD)) & (np.isin(ST_hx,ST_h)) & (np.isin(ST_wx,ST_w)))[0]

    # distribution of retirement ages
    age = np.nanmax(np.arange(par.start_T-par.ad_min,par.start_T+par.simT+par.ad_max)*
                    sim.d[idx,:,ma],axis=1)+1

    # return
    return age    

def policy_simulation(model,var,ages):
    """ policy simulation for singles"""

    if var == 'd':
        return {'hs':       lifecycle(model,var=var,MA=[0,1],ST=[1,3],ages=ages,calc='sum')['y'][0],
                'hs_f':     lifecycle(model,var=var,MA=[0],ST=[1,3],ages=ages,calc='sum')['y'][0],
                'hs_m':     lifecycle(model,var=var,MA=[1],ST=[1,3],ages=ages,calc='sum')['y'][0],
                'base':     lifecycle(model,var=var,MA=[0,1],ages=ages,calc='sum')['y'][0],
                'base_f':   lifecycle(model,var=var,MA=[0],ages=ages,calc='sum')['y'][0],
                'base_m':   lifecycle(model,var=var,MA=[1],ages=ages,calc='sum')['y'][0],
                'ls':       lifecycle(model,var=var,MA=[0,1],ST=[0,2],ages=ages,calc='sum')['y'][0],
                'ls_f':     lifecycle(model,var=var,MA=[0],ST=[0,2],ages=ages,calc='sum')['y'][0],
                'ls_m':     lifecycle(model,var=var,MA=[1],ST=[0,2],ages=ages,calc='sum')['y'][0]
        }

    if var == 'probs':
        return {'base_f':   retirement_probs(model,MA=[0]),
                'base_m':   retirement_probs(model,MA=[1])
        }

    if var == 'GovS':
        return lifecycle(model,var=var,MA=[0,1],ST=[0,1,2,3],ages=ages,calc='total_sum')['y'][0]

    if var == 'RetAge':
        return {'hs': np.mean(RetAge_S(model,ST=[1,3])),
                'base': np.mean(RetAge_S(model)),
                'ls': np.mean(RetAge_S(model,ST=[0,2]))}

def policy_simulation_c(model,var,ages):
    """ policy simulation for couples"""

    if var == 'd':
        return {'hs':       lifecycle_c(model,var=var,MA=[0],ST_w=[1,3],ages=ages,calc='sum')['y'][0] + 
                            lifecycle_c(model,var=var,MA=[1],ST_h=[1,3],ages=ages,calc='sum')['y'][0],
                'hs_f':     lifecycle_c(model,var=var,MA=[0],ST_w=[1,3],ages=ages,calc='sum')['y'][0],
                'hs_m':     lifecycle_c(model,var=var,MA=[1],ST_h=[1,3],ages=ages,calc='sum')['y'][0],
                'base':     lifecycle_c(model,var=var,MA=[0,1],ages=ages,calc='sum')['y'][0],
                'base_f':   lifecycle_c(model,var=var,MA=[0],ages=ages,calc='sum')['y'][0],
                'base_m':   lifecycle_c(model,var=var,MA=[1],ages=ages,calc='sum')['y'][0],
                'ls':       lifecycle_c(model,var=var,MA=[0],ST_w=[0,2],ages=ages,calc='sum')['y'][0] + 
                            lifecycle_c(model,var=var,MA=[1],ST_h=[0,2],ages=ages,calc='sum')['y'][0],                
                'ls_f':     lifecycle_c(model,var=var,MA=[0],ST_w=[0,2],ages=ages,calc='sum')['y'][0],
                'ls_m':     lifecycle_c(model,var=var,MA=[1],ST_h=[0,2],ages=ages,calc='sum')['y'][0]
        }

    if var == 'probs':
        return {'base_f':   retirement_probs_c(model,ma=0),
                'base_m':   retirement_probs_c(model,ma=1)
        }

    if var == 'GovS':
        return lifecycle_c(model,var=var,MA=[0,1],ages=ages,calc='total_sum')['y'][0]

    if var == 'RetAge':
        return {'hs': 
                np.mean(np.concatenate((RetAge_C(model,ma=0,ST_w=[1,3]),
                                        RetAge_C(model,ma=1,ST_h=[1,3])))),
                'base': 
                np.mean(np.concatenate((RetAge_C(model,ma=0),
                                        RetAge_C(model,ma=1)))),
                'ls': 
                np.mean(np.concatenate((RetAge_C(model,ma=0,ST_w=[0,2]),
                                        RetAge_C(model,ma=1,ST_h=[0,2]))))
                }                                         

def resolve(model,vars,recompute=True,accuracy=False,tax=True,ages=[57,110],**kwargs):
    
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
            y = policy_simulation(model,var=var,ages=ages)
            store[var].append(y)

    # return
    return store 

def resolve_c(model,vars,recompute=True,accuracy=False,tax=True,
              AD=[-4,-3,-2,-1,0,1,2,3,4],ST_h=[0,1,2,3],ST_w=[0,1,2,3],ages=[53,110],**kwargs):

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
            y = policy_simulation_c(model,var=var,ages=ages)
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