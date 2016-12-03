"""
This file is part of the Avast project.
Copyright 2016 Megan Bedell (Chicago).
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters # Hogg doesn't like lmfit
c = 2.99792458e8 # precise enough?

def make_pars(ams0, scales0, vs0):
    # given starting guesses, create a Parameters() object for fitting
    pars = Parameters()
    for i,am in enumerate(ams0):
        pars.add('ams'+str(i), value=am)
    for i,scale in enumerate(scales0):
        pars.add('scales'+str(i), value=scale)
    for i,v in enumerate(vs0):
        pars.add('vs'+str(i), value=v)
    return pars
    
def unpack_pars(pars, n_ms, n_epoch):
    # unpack Parameters() object
    ams = []
    scales = []
    vs = []
    for i in range(n_ms):
        name = 'ams'+str(i)
        ams = np.append(ams, pars[name])
    for i in range(n_epoch):
        name = 'scales'+str(i)
        scales = np.append(scales, pars[name])
    for i in range(n_epoch):
        name = 'vs'+str(i)
        vs = np.append(vs, pars[name])
    return ams, scales, vs
    
def change_par_status(pars, name, vary=False):
    # change the fitting status of a given parameter. fixes the parameter by default.
    if not name in ['ams','scales','vs']:
        print("{0} is not a valid parameter.".format(name))
        return pars
    for par_key in pars.iterkeys():
        if name in par_key:
            pars[par_key].vary = vary
    return pars

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
    pars0 = make_pars(ams0, scales0, vs0)
    pars0 = change_par_status(pars0, 'vs', vary=False)  # fix the velocities
    soln = minimize(min_function, pars0, args=fa)
    # second fit with velocities free:
    pars1 = soln.params
    pars1 = change_par_status(pars1, 'vs', vary=True)
    soln = minimize(min_function, pars1, args=fa)
    # look at the fit:
    pars = soln.params
    ams, scales, vs = unpack_pars(pars, n_ms, n_epoch)
    
    '''''
    for e in range(n_epoch):
        calc = model(xs[e], xms, del_x, ams * scales[e])
        x_plot = np.linspace(lnwave[0],lnwave[-1],num=1000)
        calc_plot = model(x_plot, xms, del_x, ams * scales[e])
        show_plot(xs[e], ys[e], calc, x_plot, calc_plot)
    '''
        
    # plotting objective function with starting RV for Hogg:
    pars2 = soln.params
    star_v1 = np.linspace(-20.,20.,500)
    obj = []  # objective function
    for v in star_v1:
        pars2['vs0'].value = v
        resids = min_function(pars2, xs, ys, xms, del_x)
        obj = np.append(obj, np.dot(resids,resids))
    plt.clf()
    plt.plot(star_v1,obj)
    plt.xlabel('starting RV guess (first epoch, m/s)')
    plt.ylabel('objective function')
    plt.savefig('objectivefn.png')

