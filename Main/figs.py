import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets

def cons_time(model,t):
    
    # convert to list
    if type(t) == int:
        t = [t]
    
    # a. unpack
    par = model.par
    sol = model.sol
    poc = par.poc
    
    # b. loop
    for i in t:
        
        m = sol.m[i]
        c = sol.c[i]

        # c. figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # d. plot consumption
        ax.plot(m[:,0],c[:,0],label='$C_t(d_t=0)$')
        ax.plot(m[:,1],c[:,1],label='$C_t(d_t=1)$')
        ax.set_title(f'($t = {i}$)',pad=10)

        # f. details
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('$m_t$')
        #ax.set_xlim(min(m[poc,0],m[0,1]),max(m[-1,0], m[-1,1]))
        ax.set_ylabel('$C_t$')
        #ax.set_ylim(min(c[poc,0],c[0,1]),max(c[-1,0], c[-1,1]))

        plt.show()

def cons_choice(model,t,st,choice='work'):
    
    # a. unpack
    par = model.par
    sol = model.sol
    poc = par.poc
            
    # extract right variables
    if choice=='work':
        m = sol.m[:,st,:,1]
        c = sol.c[:,st,:,1]
    else:
        m = sol.m[:,st,:,0]
        c = sol.c[:,st,:,0]        
            
    # convert to list
    if type(t) == int:
        t = [t]
        
    # c. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
        
    # d. plot consumption
    for i in t:
        ax.plot(m[i],c[i],label=(f'($t = {i}$)'))  
        
    if choice=='work':
        ax.set_title('working')
    else:
        ax.set_title('retired')        
        
    # f. details
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    #ax.set_xlim(0,15)
    ax.set_ylabel('$C_t$')
    #ax.set_ylim(0,10)
        
    plt.show()

def consumption_function(model,t):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')

    p,m = np.meshgrid(par.grid_p, par.grid_m,indexing='ij')

    # c. plot consumption
    ax.plot_surface(p,m,sol.c[t,:,:],edgecolor='none',cmap=cm.viridis)
    ax.set_title(f'$c$ ($t = {t})$',pad=10)

    # d. details
    ax.grid(True)
    ax.set_xlabel('$p_t$')
    ax.set_xlim([par.grid_p[0],par.grid_p[-1]])
    ax.set_ylabel('$m_t$')
    ax.set_ylim([par.grid_m[0],par.grid_m[-1]])
    ax.invert_xaxis()

    plt.show()

def consumption_function_interact(model):

    widgets.interact(consumption_function,
        model=widgets.fixed(model),
        t=widgets.Dropdown(description='t', 
            options=list(range(model.par.T)), value=0),
        )          

def lifecycle(model,vars=['m','c','a']):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure()

    simvardict = dict([('m','$m_t$'),
                  ('c','$c_t$'),
                  ('a','$a_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    age = np.arange(110-par.T+1,110+1)
    ax = fig.add_subplot(1,1,1)
    
    for i in vars:
        simdata = getattr(sim,i)
        ax.plot(age,np.nanmean(simdata,axis=1),lw=2,label=simvardict[i])
    
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('age')
    if ('m' in vars or 'c' in vars or 'a' in vars):
        ax.set_ylabel('100.000 DKR')
