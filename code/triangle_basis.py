import numpy as np
from numpy import log
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def triangle(x, m, del_x):
    # returns value of triangle component centered on m at location x
    return np.maximum(1. - np.abs(x - m)/del_x, 0.)
    
def model(x, m, del_x, a_m):
    # returns value of triangle-based model at x
    # x : ln(wavelength) at point of interest
    # m : ln(wavelength) grid, shape (M)
    # del_x : ln(wavelength) spacing
    # a_m : function coefficients, shape (M)
    return np.sum(a_m*triangle(x, m, del_x)) 

def min_function(a_m, xs, ys, m, del_x):
    # function to minimize with mpfit
    calc = [model(x, m, del_x, a_m) for x in xs]
    err = np.sqrt(ys)    # assumes Poisson noise 
    return (ys - calc)/err
    
def show_plot(x, obs, calc):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(x,obs, color='black')
    ax1.plot(x,calc, color='red')
    ax.set_xticklabels( () )
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.plot(x,obs - calc, color='black')
    fig.subplots_adjust(hspace=0.05)
    plt.show()
  

if __name__ == "__main__":
    wave, spec = np.loadtxt('../test_spec.txt', unpack=True)
    lnwave = log(wave)
    m = np.linspace(lnwave[0], lnwave[-1], num=50)
    del_x = (lnwave[-1]-lnwave[0])/50.
    
    fa = (lnwave, spec, m, del_x)
    soln = least_squares(min_function, np.ones(50), args=fa)
    a_m = soln.x
    calc = [model(w, m, del_x, a_m) for w in lnwave]
    
    show_plot(lnwave, spec, calc)