"""
This file is part of the Avast project.
Copyright 2017 Megan Bedell (Chicago), David W. Hogg (NYU), and Dan Foreman-Mackey (Washington).
"""
import celerite
from celerite import terms
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def shift_and_flatten(xis, data):
    ndata = sum([len(d[0]) for d in data])
    ndata_byepoch = [len(d[0]) for d in data]
    n = 0
    x = np.empty(ndata)
    y = np.empty(ndata)
    yerr = np.empty(ndata)
    for i, d in enumerate(data):
        length = len(d[0])
        x[n:n+length] = d[0] - xis[i]
        y[n:n+length] = d[1]
        yerr[n:n+length] = d[2]
        n += length
    return x, y, yerr

def set_params(params, data):
    ndata = sum([len(d[0]) for d in data])
    nepoch = len(subset)
    ndata_byepoch = [len(d[0]) for d in data]
    xis, gp_par = params[0:len(data)], params[len(data):]
    x, y, yerr = shift_and_flatten(xis, data)
    inds = np.argsort(x)
    x = x[inds]
    y = y[inds]
    yerr = yerr[inds]
    gp.set_parameter_vector(gp_par)
    gp.compute(x, yerr)
    eye = np.eye(nepoch)
    design = np.repeat(eye, ndata_byepoch, axis=0)
    A = design[inds,:]
    scales = np.linalg.solve(np.dot(A.T, gp.apply_inverse(A)), np.dot(A.T,gp.apply_inverse(y)))
    y[np.argsort(inds)] -= np.repeat(scales, ndata_byepoch)
    return scales, xis, y
    
def nll(params, data):
    scales, xis, y = set_params(params, data)
    return -gp.log_likelihood(y) + 1./2. #* np.sum(xis**2)
        
def prediction(params, data):
    scales, xis, y = set_params(params, data)
    result_flat = gp.predict(y, return_cov=False)
    x, _, _ = shift_and_flatten(xis, data)
    inds = np.argsort(x)
    result_sorted = result_flat[np.argsort(inds)]
    result = []
    n = 0
    for i,d in enumerate(data):
        length = len(d[0])
        result.append(result_sorted[n:n+length] + scales[i])
        n += length
    return result
    
if __name__ == "__main__":
    data = pickle.load(open( "data.p", "rb" ))
    
    print "Reading in data..."
    wave_lo = np.log(4680.)
    wave_hi = np.log(4700.)
    subset = []
    #for i in range(len(data)):
    for i in range(0,len(data),10):
        m = (data[i][0] > wave_lo) & (data[i][0] < wave_hi)
        x = np.copy(data[i][0][m])
        y = np.log(np.copy(data[i][1][m]))
        yerr = np.copy(data[i][2][m]/data[i][1][m])
        subset.append((x,y,yerr))
        
    print "Data read."    
    kernel = terms.RealTerm(0.04, -np.log(0.001), bounds=((.01*0.04,100.*0.04),(None, None)))
    gp = celerite.GP(kernel,
                     log_white_noise=-9.6,
                     fit_white_noise=True)
                     
    print "Minimizing..."                                      
    xis0 = np.zeros(len(subset))
    p0 = np.append(xis0, gp.get_parameter_vector())
    bounds = [(-1e-3, 1e-3) for d in subset] + gp.get_parameter_bounds()
    soln = minimize(nll, p0, args=(subset), bounds=bounds, method='L-BFGS-B')
    scales, xis, y = set_params(soln.x, subset)
    
    '''''
    fig,ax = plt.subplots(1,1,figsize=(12,4))
    for i,d in enumerate(subset):
        ax.plot(d[0] - xis[i],d[1] - scales[i])
    ax.set_xlabel('ln(wavelength)')
    ax.set_ylabel('ln(flux) - scale factor')
    '''
    
    print "Calculating model prediction..."
    mu = prediction(soln.x, subset)
    
    fig,ax = plt.subplots(1,1,figsize=(12,4))
    for i,d in enumerate(subset):
        ax.plot(d[0], d[1], color='black')
        ax.plot(d[0], mu[i], color='red')
    ax.set_xlabel('ln(wavelength)')
    ax.set_ylabel('ln(flux) - scale factor')
        
    fig,ax = plt.subplots(1,1,figsize=(12,4))
    for i,d in enumerate(subset):
        ax.plot(d[0], (np.exp(d[1]) - np.exp(mu[i])) + 1000*i, color='black')
    ax.set_xlabel('ln(wavelength)')
    ax.set_ylabel('(O - C) + offset')
    