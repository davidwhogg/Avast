"""
This file is part of the Avast project.
Copyright 2016 Megan Bedell (Chicago) and David W. Hogg (NYU).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import leastsq, fmin_bfgs
c = 2.99792458e8   # m/s

def deriv_test(x, f, dfdx, ind, step, args):
    assert step > 0.
    y1 = f(x, *args)
    x2 = x.copy()
    x2[ind] += step
    y2 = f(x2, *args)
    dfdx2 = (y2 - y1)/step
    dfdx0 = (dfdx(x, *args))[ind]
    return (dfdx2 - dfdx0)/dfdx0
    
def vector_deriv_test(x, f, dfdx, ind, step, args):
    assert step > 0.
    y1 = f(x, *args)
    x2 = x.copy()
    x2[ind] += step
    y2 = f(x2, *args)
    dfdx2 = (y2 - y1)/step
    dfdx0 = (dfdx(x, *args))[:,ind]
    return (dfdx2 - dfdx0) / (dfdx2 + dfdx0)