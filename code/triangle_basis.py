"""
This file is part of the Avast project.
Copyright 2016 Megan Bedell (Chicago).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
c = 2.99792458e8

def unpack_pars(pars, n_ms, n_epoch):
    # unpack Parameters() object
    ams = pars[0:n_ms]
    scales = pars[n_ms:n_ms+n_epoch]
    vs = pars[n_ms+n_epoch:]
    return ams, scales, vs
    
def triangle(xs, xms, del_x):
    # returns values of triangle components centered on xms at location xs
    # xs & xms must be broadcastable
    return np.maximum(1. - np.abs(xs - xms)/del_x, 0.)
    
def model(xs, xms, del_x, ams):
    # returns values of triangle-based model at xs
    # xs : ln(wavelength) at point(s) of interest
    # xms : ln(wavelength) grid, shape (M)
    # del_x : ln(wavelength) spacing of xms
    # ams : function coefficients, shape (M)
    return np.sum(ams[None,:] * triangle(xs[:,None], xms[None,:], del_x), axis=1) 

def min_function(pars, xs, ys, xms, del_x):
    # function to minimize
    n_epoch = len(xs)
    n_ms = len(xms)
    ams, scales, vs = unpack_pars(pars, n_ms, n_epoch)
    resid = np.array([])
    for e in range(n_epoch):
        beta = vs[e] / c
        thisxs = xs[e] - 0.5 * np.log((1. + beta)/(1 - beta))
        calc = model(thisxs, xms, del_x, ams * scales[e])
        err = np.sqrt(ys[e])    # assumes Poisson noise 
        resid = np.append(resid,(ys[e] - calc) / err)
    return np.append(resid, np.append((scales - 1.) / 0.5, (vs - 0.) / 30.)) #MAGIC
    
def show_plot(xs, obs, calc, x_plot, calc_plot):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.step(xs,obs, color='black')
    ax1.plot(x_plot,calc_plot, color='red')
    #ax1.set_xticklabels( () )
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.step(xs,obs - calc, color='black')
    fig.subplots_adjust(hspace=0.05)
    plt.show()

if __name__ == "__main__":
    wave, spec = np.loadtxt('../data/test_spec1.txt', unpack=True)
    wave2, spec2 = np.loadtxt('../data/test_spec2.txt', unpack=True)
    wave3, spec3 = np.loadtxt('../data/test_spec3.txt', unpack=True)
    lnwave = np.log(wave)
    lnwave2 = np.log(wave2)
    lnwave3 = np.log(wave3)
    xs = [lnwave, lnwave2, lnwave3]
    ys = [spec, spec2, spec3]
    del_x = 1.3e-5/2.0
    xms = np.arange(np.min(lnwave) - 0.5 * del_x, np.max(lnwave) + 0.99 * del_x, del_x)
    
    # initial fit to ams & scales:
    fa = (xs, ys, xms, del_x)
    n_epoch = len(xs)
    n_ms = len(xms)
    ams0 = np.random.normal(size=n_ms) + np.median(ys[0])
    scales0 = np.ones(n_epoch)
    vs0 = np.random.normal(size=n_epoch) * 5.0
    pars0 = np.append(ams0, np.append(scales0, vs0))
    ftol = 1.49012e-08  # default is 1.49012e-08
    soln = leastsq(min_function, pars0, args=fa, ftol=ftol)

    # look at the fit:
    pars = soln[0]
    ams, scales, vs = unpack_pars(pars, n_ms, n_epoch)
    

    for e in range(n_epoch):
        calc = model(xs[e], xms, del_x, ams * scales[e])
        x_plot = np.linspace(lnwave[0],lnwave[-1],num=1000)
        calc_plot = model(x_plot, xms, del_x, ams * scales[e])
        show_plot(xs[e], ys[e], calc, x_plot, calc_plot)

     
    # plotting objective function with leastsq tolerance:
    obj = []  # objective function
    all_ftol = []
    for i in range(10):
        soln = leastsq(min_function, pars, args=fa, ftol=ftol)
        resids = min_function(soln[0], xs, ys, xms, del_x)
        obj = np.append(obj, np.dot(resids,resids))
        ftol = ftol/10.  # tighten tolerance
        all_ftol = np.append(all_ftol, ftol)
    plt.clf()
    plt.plot(all_ftol,obj)
    plt.xscale('log')
    plt.xlabel('ftol')
    plt.ylabel('objective function')
    plt.savefig('objectivefn_ftol.png')

