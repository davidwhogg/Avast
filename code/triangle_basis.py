import numpy as np
from numpy import log
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pdb
c = 3.e8

def triangle(xs, xms, del_x):
    # returns value of triangle component centered on m at location x
    # xs & xms must be broadcastable
    return np.maximum(1. - np.abs(xs - xms)/del_x, 0.)
    
def model(xs, xms, del_x, ams):
    # returns value of triangle-based model at x
    # x : ln(wavelength) at point of interest
    # m : ln(wavelength) grid, shape (M)
    # del_x : ln(wavelength) spacing
    # ams : function coefficients, shape (M)
    return np.sum(ams[None,:] * triangle(xs[:,None], xms[None,:], del_x), axis=1) 

def unpack_pars(pars, nspec):
    return pars[:-2 * nspec], pars[-2 * nspec:-nspec], pars[-nspec:]

def min_function(pars, xs, ys, xms, del_x):
    # function to minimize with mpfit
    nepoch = len(xs)
    ams, scales, vs = unpack_pars(pars, nepoch)
    resid = np.array([])
    for e in range(nepoch):
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
    ax1.set_xticklabels( () )
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.step(xs,obs - calc, color='black')
    fig.subplots_adjust(hspace=0.05)
    plt.show()
  
if __name__ == "__main__":
    wave, spec = np.loadtxt('../test_spec1.txt', unpack=True)
    #wave2, spec2 = np.loadtxt('../test_spec2.txt', unpack=True)
    wave3, spec3 = np.loadtxt('../test_spec3.txt', unpack=True)
    lnwave = log(wave)
    #lnwave2 = log(wave2)
    lnwave3 = log(wave3)
    xs = [lnwave, lnwave3]
    ys = [spec, spec3]
    del_x = 1.3e-5/2.0
    xms = np.arange(np.min(lnwave) - 0.5 * del_x, np.max(lnwave) + 0.99 * del_x, del_x)
    
    fa = (xs, ys, xms, del_x)
    nepoch = len(xs)
    ams0 = np.random.normal(size=len(xms)) + np.median(ys[0])
    vs0 = 5. * np.random.normal(size=nepoch)
    pars0 = np.append(ams0, np.append(np.ones(nepoch), vs0))
    foo, bar, what = unpack_pars(pars0, nepoch)
    soln = least_squares(min_function, pars0, args=fa)
    pars = soln.x
    ams, scales, vs = unpack_pars(pars, nepoch)
    for e in range(nepoch):
        calc = model(xs[e], xms, del_x, ams * scales[e])
        x_plot = np.linspace(lnwave[0],lnwave[-1],num=1000)
        calc_plot = model(x_plot, xms, del_x, ams * scales[e])
        show_plot(xs[e], ys[e], calc, x_plot, calc_plot)
        