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
from scipy.linalg import svd
from scipy.io.idl import readsav
c = 2.99792458e8   # m/s

def shift_and_flatten(xis, data):
    # applies shifts of xi to the data and returns flattened data arrays
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
    # update the GP parameter vector & optimize scales
    # returns: scales and xi parameters used, sorted and scaled data vector y
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
    # negative ln(likelihood) function for optimization
    scales, xis, y = set_params(params, data)
    return -gp.log_likelihood(y) + 1./2. #* np.sum(xis**2)
        
def prediction(params, data):
    # returns the model predicted by params in the same shape as data
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
    
def xi_to_v(xi):
    # translate ln(wavelength) Doppler shift to a velocity in m/s
    return np.tanh(xi) * c
 
def v_to_xi(v):
    return np.arctanh(v/c)
    
if __name__ == "__main__":
    
    print "Reading in data..."
    data = pickle.load(open( "data.p", "rb" ))
    wave_lo = np.log(6553.)
    wave_hi = np.log(6573.)
    subset = []
    subsample = 5 # int between 1 and len(data), smaller selects more epochs
    for i in range(0,len(data),subsample):
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
    
    # (optional) initialize xis using the HARPS pipeline RVs:                 
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    pipeline = readsav(data_dir+'HIP22263_result.dat') 
    xis0 = np.empty(len(subset))
    rvs = np.empty(len(subset))
    dates = np.empty(len(subset))
    shks = np.empty(len(subset))
    for i in range(len(subset)):
        rvs[i] = pipeline.rv[i*subsample] * 1.e3
        xis0[i] = v_to_xi(rvs[i])
        dates[i] = pipeline.date[i*subsample]
        shks[i] = pipeline.shk[i*subsample]
    
    #xis0 = np.zeros(len(subset))
    p0 = np.append(xis0, gp.get_parameter_vector())
    xi_bounds = [(-1e-16+xi, 1e-16+xi) for xi in xis0] # fixed RVs
    bounds = xi_bounds + gp.get_parameter_bounds()
    soln = minimize(nll, p0, args=(subset), bounds=bounds, method='L-BFGS-B')
    scales, xis, y = set_params(soln.x, subset)
    
    print "RVs =", xi_to_v(xis)
    
    '''''
    fig,ax = plt.subplots(1,1,figsize=(12,4))
    for i,d in enumerate(subset):
        ax.plot(d[0] - xis[i],d[1] - scales[i])
    ax.set_xlabel('ln(wavelength)')
    ax.set_ylabel('ln(flux) - scale factor')
    '''
    
    print "Calculating model prediction..."
    mu = prediction(soln.x, subset)
    
    if False:
        # make some plots    
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
    
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,8))
        d = subset[0]
        ax1.plot(np.exp(d[0]), np.exp(d[1]), color='black')
        ax1.plot(np.exp(d[0]), np.exp(mu[0]), color='red')
        ax2.plot(np.exp(d[0]), (np.exp(d[1]) - np.exp(mu[0])), color='black')
        ax1.set_xlim(np.exp([wave_lo,wave_hi]))
        ax2.set_xlim(np.exp([wave_lo,wave_hi]))
        ax2.set_xlabel(r'Wavelength $(\AA)$')
        ax2.set_ylabel('(O - C)')
        ax1.set_ylabel('Flux')
    
    if True:
        # PCA time!
        # ONLY WORKS IF ALL SPECTRA ARE THE SAME LENGTH
        scaled_resids = np.empty((len(subset),len(subset[0][0])))
        for i,s in enumerate(subset):
            scaled_resids[i,:] = np.exp(s[1]/scales[i]) - np.exp(mu[i]/scales[i])
        u, s, v = svd(scaled_resids, full_matrices=False)
        u.shape, s.shape, v.shape
        
        plt.scatter(rvs-np.median(rvs), u[:,0], label='PCA 0')
        plt.scatter(rvs-np.median(rvs), u[:,1], label='PCA 1')
        plt.xlabel('(Relative) Pipeline RV (m/s)')
        plt.ylabel('PCA Component Strength')
        plt.legend()
        plt.savefig('fig/gp_halpha_rvpca.png')
        plt.clf()
    
        plt.scatter(shks, u[:,0], label='PCA 0')
        plt.scatter(shks, u[:,1], label='PCA 1')
        plt.xlabel('SHK Index')
        plt.ylabel('PCA Component Strength')
        plt.legend()
        plt.savefig('fig/gp_halpha_shkpca.png')
        plt.clf()
        
        wave = np.exp(subset[0][0])
        plt.plot(wave, v[0,:], label='PCA 0')
        plt.plot(wave, v[1,:], label='PCA 1')
        plt.plot(wave, v[2,:], label='PCA 2')
        plt.plot(wave, v[3,:], label='PCA 3')
        plt.legend()
        plt.savefig('fig/gp_halpha_pca.png')
        
        
    