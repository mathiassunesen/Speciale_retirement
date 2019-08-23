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

def policy(model,ax,time,policy_type='c',d_choice=[0,1],states=[0],xlim=None):

    # convert to list
    if type(time) == int:
        time = [time] 
    if type(d_choice) == int:
        d_choice = [d_choice]
    if type(states) == int:
        states == [states]

    # unpack
    sol = model.sol

    # b. policy type
    if policy_type == 'c':
        policy = sol.c[:,:,:,:,0] # zero in the end to get the main sol
        ylab = '$C_t$'
    elif policy_type == 'v':
        policy = sol.v[:,:,:,:,0]
        ylab = '$v_t$'
    m = sol.m[:,:,:,:,0]

    # c. states
    if states == 'all':
        states = np.arange(len(model.par.states))
    
    # d. loop over time
    for t in time:
        for st in states:   
            for d in d_choice:
                # plot
                if policy_type == 'c':
                    lab = '$C_t(t = {}, d = {})$'.format(t,d)
                elif policy_type == 'v':
                    lab = '$v_t(t = {}, d = {})$'.format(t,d)
                ax.plot(m[t,st,:,d], policy[t,st,:,d], label=lab)

    # e. details
    if xlim == None:
        pass
    else:
        ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel(ylab)

def lifecycle(model,ax,vars=['m','c','a'],ages=[57,68]):

    # a. unpack
    sim = model.sim

    # b. figure
    simvardict = dict([('m','$m_t$'),
                  ('c','$c_t$'),
                  ('a','$a_t$'),
                  ('d','$d_t$'),
                  ('alive','$alive_t$')])

    x = np.arange(ages[0], ages[1]+1)
    for i in vars:
        simdata = getattr(sim,i)[transitions.inv_age(x)]
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
        

    
