"""
This file is part of the Avast project.
Copyright 2016 Megan Bedell (Chicago) and David W. Hogg (NYU).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import leastsq
c = 2.99792458e8   # m/s

def unpack_pars(pars, n_ms, n_epoch):
    # unpack parameters
    ams = pars[0:n_ms]
    scales = pars[n_ms:n_ms+n_epoch]
    xis = pars[n_ms+n_epoch:]
    return ams, scales, xis

def xi_to_v(xi):
    # translate ln(wavelength) Doppler shift to a velocity in m/s
    r = np.e**(2.*xi) # r = (c+v)/(c-v)
    return c * (r - 1.)/(r + 1.)
    
def g(xs, xms, del_x):
    # returns values of triangle components centered on xms at location xs
    # xs & xms must be broadcastable
    return np.maximum(1. - np.abs(xs - xms)/del_x, 0.)

def dg_dx(xs, xms, del_x):
    # return the derivatives of `g()`
    signs = np.sign(xms - xs)
    signs[np.abs(xms - xs) > del_x] = 0.
    return signs/del_x

def f(xs, xms, del_x, ams):
    # returns values of triangle-based model f(x) at xs
    # xs : ln(wavelength) at point(s) of interest
    # xms : ln(wavelength) grid, shape (M)
    # del_x : ln(wavelength) spacing of xms
    # ams : function coefficients, shape (M)
    return np.sum(ams[None,:] * g(xs[:,None], xms[None,:], del_x), axis=1) 

def min_function(pars, xs, ys, xms, del_x):
    # function to minimize
    n_epoch = len(xs)
    n_ms = len(xms)
    ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
    resid = np.array([])
    for e in range(n_epoch):
        xprimes = xs[e] + xis[e]
        calc = scales[e] * f(xprimes, xms, del_x, ams)
        err = np.sqrt(ys[e])    # assumes Poisson noise 
        resid = np.append(resid,(ys[e] - calc) / err)
    return np.append(resid, np.append((scales - 1.) / 0.5)) #MAGIC

def deriv_function(pars, xs, ys, xms, del_x):
    # derivatives of min_function() wrt pars
    n_epoch = len(xs)
    n_ms = len(xms)
    ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
    deriv_matrix = np.zeros((len(xs[0]), n_epoch, len(pars)))
    for j in range(n_epoch):
        xprimes = xs[j] + xis[e]
        dy_dams = scales[j] * g(xprimes[:,None], xms[None,:], del_x)
        deriv_matrix[:,j,:n_ms] = dy_dams
        dy_dsj = f(xprimes, xms, del_x, ams)
        deriv_matrix[:,j,n_ms+j] = dy_dsj
        dy_dxij = scales[j] * np.sum(ams[None,:] * dg_dx(xprimes[:,None], 
            xms[None,:], del_x), axis=1)
        deriv_matrix[:,j,-n_epoch+j] = dy_dxij
    return deriv_matrix
    
def min_v(pars, i, xs, ys, xms, del_x):
    # do a simple minimization of just one xi parameter
    # i : epoch # to minimize, 0-2
    tmp_pars = np.copy(pars)    
    obj = []  # objective function
    xi0 = []
    for xi in np.linspace(xis[i]-1.,xis[i]+1.,100):
        tmp_pars[-n_epoch+i] = xi
        resids = min_function(tmp_pars, xs, ys, xms, del_x)
        obj = np.append(obj, np.dot(resids,resids))
        xi0 = np.append(xi0,xi)
    plt.clf()
    plt.plot(xi_to_v(xi0),obj)
    plt.axvline(xi_to_v(xis[i]))
    plt.xlabel('v (m/s)')
    plt.ylabel('objective function')
    plt.savefig('objectivefn_v{0}.png'.format(i))
    xi_min = xi0[np.argmin(obj)]
    tmp_pars[-n_epoch+i] = xi_min
    return tmp_pars
    
def save_plot(xs, obs, calc, x_plot, calc_plot, save_name):
    xs = np.e**xs
    x_plot = np.e**x_plot
    fig = plt.figure()

    ax1 = fig.add_subplot(2,1,1)
    ax1.step(xs,obs, color='black', label='Observed')
    ax1.plot(x_plot,calc_plot, color='red', label='Calculated')
    ax1.set_ylabel('Flux')
    #ax1.legend()
    ax1.set_xticklabels( () )
    ax2 = fig.add_subplot(2,1,2)
    ax2.step(xs,obs - calc, color='black')
    ax2.set_ylabel('(O-C)')
    ax2.ticklabel_format(useOffset=False)
    ax2.set_xlabel(r'Wavelength ($\AA$)')
    majorLocator = MultipleLocator(1)
    minorLocator = MultipleLocator(0.1)
    ax1.xaxis.set_minor_locator(minorLocator)
    ax1.xaxis.set_major_locator(majorLocator)
    ax2.xaxis.set_minor_locator(minorLocator)
    ax2.xaxis.set_major_locator(majorLocator)
    majorLocator = MultipleLocator(5000)
    ax1.yaxis.set_major_locator(majorLocator)
    majorLocator = MultipleLocator(200)
    ax2.yaxis.set_major_locator(majorLocator)
    ax2.set_ylim([-500,500])
    fig.subplots_adjust(hspace=0.05)
    plt.savefig(save_name)

if __name__ == "__main__":
    data_dir = '../data/binary_star/'
    wave, spec = np.loadtxt(data_dir+'test_spec1.txt', unpack=True)
    wave2, spec2 = np.loadtxt(data_dir+'test_spec2.txt', unpack=True)
    wave3, spec3 = np.loadtxt(data_dir+'test_spec3.txt', unpack=True)
    wave4, spec4 = np.loadtxt(data_dir+'test_spec4.txt', unpack=True)
    wave5, spec5 = np.loadtxt(data_dir+'test_spec5.txt', unpack=True)
    xs = np.log([wave, wave2, wave3, wave4, wave5])
    ys = [spec, spec2, spec3, spec4, spec5]
    del_x = 1.3e-5/2.0
    xms = np.arange(np.min(xs) - 0.5 * del_x, np.max(xs) + 0.99 * del_x, del_x)
    
    # initial fit to ams & scales:
    fa = (xs, ys, xms, del_x)
    n_epoch = len(xs)
    n_ms = len(xms)
    ams0 = np.random.normal(size=n_ms) + np.median(ys[0])
    scales0 = np.ones(n_epoch)
    xis0 = np.random.normal(size=n_epoch)
    pars0 = np.append(ams0, np.append(scales0, xis0))
    ftol = 1.49012e-08  # default is 1.49012e-08
    soln = leastsq(min_function, pars0, args=fa, ftol=ftol, full_output=True)

    # look at the fit:
    pars = soln[0]
    ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
    resids = min_function(pars, xs, ys, xms, del_x)
    print "Initial optimization of all parameters:"
    print "Objective function value: {0}".format(np.dot(resids,resids))
    print "nfev: {0}".format(soln[2]['nfev'])
    print "mesg: {0}".format(soln[3])
    print "ier: {0}".format(soln[4])
    #vs = xi_to_v(xis)
    #print "Velocities:", vs

    
    # optimize one epoch at a time:
    for i in range(n_epoch):
        pars = min_v(pars, i, xs, ys, xms, del_x)
        resids = min_function(pars, xs, ys, xms, del_x)
        ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
        print "Optimization of velocity at epoch {0}:".format(i)
        print "Objective function value: {0}".format(np.dot(resids,resids))
        #vs = xi_to_v(xis)
        #print "Velocities:", vs
