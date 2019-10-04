import numpy as np
import warnings

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

# local modules
import transitions

def policy(model,ax,var,time,states,age_dif,d_choice,xlim=None):

    # a. convert to list
    if type(time) == int:
        time = [time]
    if type(states) == int:
        states = [states] 
    if type(age_dif) == int:
        age_dif = [age_dif]         
    if type(d_choice) == int:
        d_choice = [d_choice]

    # b. unpack
    sol = model.sol

    # c. figure
    solvardict = dict([('c','C_t'),
                       ('v','v_t')])    
    m = sol.m
    
    # c. loop through options
    for t in time:
        for st in states:
            for ad in age_dif:
                for d in d_choice:

                    x = m[t,st,ad,:,d]
                    y = getattr(sol,var)[t,st,ad,:,d]
                    lab = '${}(t = {}, d = {})$'.format(solvardict[var],t,d)
                    ax.plot(x,y,label=lab)

    # e. details
    if xlim == None:
        pass
    else:
        ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('${}$'.format(solvardict[var]))

def lifecycle(model,ax,vars=['m','c','a'],ages=[57,68]):

    # a. unpack
    sim = model.sim
    par = model.par

    # b. figure
    simvardict = dict([('m','$m_t$'),
                  ('c','$C_t$'),
                  ('a','$a_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    x = np.arange(ages[0], ages[1]+1)
    for i in vars:
        simdata = getattr(sim,i)[transitions.inv_age(x,par)]
        with warnings.catch_warnings(): # ignore this specific warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ax.plot(x,np.nanmean(simdata,axis=1),lw=2,label=simvardict[i])
    
    # c. details
    ax.legend()
    ax.grid(True)    
    ax.set_xlabel('Age')
    if (len(x) < 15):
        ax.set_xticks(x)
    if ('m' in vars or 'c' in vars or 'a' in vars):
        ax.set_ylabel('100.000 DKR')


def retirement_probs(model,ax,ages=[57,68]):
    
    # a. unpack
    sim = model.sim
    
    # b. figure
    avg_probs = np.zeros(ages[1] - ages[0]+1)
    for t in range(len(avg_probs)):
        avg_probs[t] = np.mean(sim.probs[t])

    x = np.arange(ages[0], ages[1]+1)
    ax.plot(x,avg_probs,'r')
    
    # c. details
    ax.grid(True)    
    ax.set_xticks(x)
    ax.set_ylim(top=0.35)
    ax.set_xlabel('Age')
    ax.set_ylabel('Retirement probability')
        

    
