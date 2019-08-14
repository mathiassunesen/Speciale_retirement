import numpy as np
from numba import njit

def GaussHermite(n):
    return np.polynomial.hermite.hermgauss(n)

def GaussHermite_lognorm(sigma, n):
    x,w = GaussHermite(n)

    x = np.exp(x*np.sqrt(2)*sigma - 0.5*sigma**2)
    w = w/np.sqrt(np.pi)

    assert(1 - sum(w*x) < 1e-8)
    return x,w    

def logsum(v1, v2, sigma):

    # 1. setup
    V = np.stack((v1,v2), axis=1)

    # 2. maximum over the discrete choices
    mxm = np.amax(V, axis=1, keepdims=True)

    # 3. logsum and probabilities
    if abs(sigma) > 1e-10:
        log_sum = mxm + sigma*np.log(np.sum(np.exp((V - mxm @ np.ones((1,2)))/sigma), 
            axis=1, keepdims=True))
        prob = np.exp((V - log_sum*np.ones((1,2)))/sigma)

    else:
        log_sum = mxm
        prob = np.zeros_like(V)
        prob[np.arange(len(V)), np.argmax(V, axis=1)] = 1

    return log_sum,prob

@njit(parallel=True)
def logsum_vec(V, par): # supports only 2 columns in order to be implemented in numba
    
    # 1. setup
    sigma = par.sigma_eta
    cols = V.shape[1]
    
    # 2. maximum over the discrete choices
    mxm = np.maximum(V[:,0], V[:,1]).reshape(len(V),1)

    # 3. logsum and probabilities
    if abs(sigma) > 1e-10:
        logsum = mxm + sigma*np.log(np.sum(np.exp((V - mxm*np.ones((1,cols)))/sigma), 
                                            axis=1)).reshape(mxm.shape)
        prob = np.exp((V - logsum*np.ones((1,cols)))/sigma)

    else:
        logsum = mxm
        prob = np.zeros_like(V)
        for i in range(len(V)):
            if V[i,0] > V[i,1]:
                prob[i,0] = 1
            else:
                prob[i,1] = 1 

    return logsum,prob