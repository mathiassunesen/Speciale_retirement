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

def policy(model,ax,time,policy_type='c',d_choice=[0,1],states=[0]):

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
        lab = '$C_t(t=0, d=1)$'
        ylab = '$C_t$'
    elif policy_type == 'v':
        policy = sol.v[:,:,:,:,0]
        lab = '$v_t$'
        ylab = '$v_t$'
    m = sol.m[:,:,:,:,0]

    # c. states
    if states == 'all':
        states = np.arange(len(model.par.states))
    
    # c. loop over time
    for t in time:
        for st in states:   
            for d in d_choice:
                # plot
                ax.plot(m[t,st,:,d], policy[t,st,:,d], label=lab)

    #ax.set_title(f'($t = {i}$)',pad=10)
    # d. details
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel(ylab)

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
    
    ax.grid(True)    
    ax.set_xticks(x)
    ax.set_ylim(top=0.35)
    ax.set_xlabel('Age')
    ax.set_ylabel('Retirement probability')
        

    
