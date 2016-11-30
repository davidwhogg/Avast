import numpy as np
from numpy import log
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pdb

def triangle(x, m, del_x):
    # returns value of triangle component centered on m at location x
    return np.maximum(1. - np.abs(x - m)/del_x, 0.)
    
def model(xs, xms, del_x, ams):
    # returns value of triangle-based model at x
    # x : ln(wavelength) at point of interest
    # m : ln(wavelength) grid, shape (M)
    # del_x : ln(wavelength) spacing
    # a_m : function coefficients, shape (M)
    return np.sum(ams[None,:] * triangle(xs[:,None], xms[None,:], del_x), axis=1) 

def min_function(a_m, xs, ys, m, del_x):
    # function to minimize with mpfit
    calc = model(xs, m, del_x, a_m)
    err = np.sqrt(ys)    # assumes Poisson noise 
    return (ys - calc)/err
    
def show_plot(x, obs, calc, x_plot, calc_plot):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.step(x,obs, color='black')
    ax1.plot(x_plot,calc_plot, color='red')
    ax1.set_xticklabels( () )
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.step(x,obs - calc, color='black')
    fig.subplots_adjust(hspace=0.05)
    plt.show()
  

if __name__ == "__main__":
    wave, spec = np.loadtxt('../test_spec.txt', unpack=True)
    lnwave = log(wave)
    del_x = 1.3e-5/2.0
    x_m = np.arange(np.min(lnwave), np.max(lnwave), del_x)
    
    fa = (lnwave, spec, x_m, del_x)
    soln = least_squares(min_function, np.ones_like(x_m), args=fa)
    a_m = soln.x
    calc = model(lnwave, x_m, del_x, a_m)
    
    x_plot = np.linspace(lnwave[0],lnwave[-1],num=1000)
    calc_plot = model(x_plot, x_m, del_x, a_m)
    
    show_plot(lnwave, spec, calc, x_plot, calc_plot)