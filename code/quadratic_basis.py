"""
This file is part of the Avast project.
Copyright 2017 David W. Hogg (NYU).
"""

import numpy as np

def g(xs_input, xms, del_x):
    """
    quadratic spline
    """
    xs = np.atleast_1d((xs_input - xms) / del_x + 1.5) # 1.5 for re-center
    ys = np.zeros_like(xs)
    I = (xs > 0.) * (xs < 1.) 
    ys[I] = 0.5 * xs[I] * xs[I]
    I = (xs > 1.) * (xs < 2.)
    ys[I] = -1. * xs[I] * xs[I] + 3. * xs[I] - 1.5
    I = (xs > 2.) * (xs < 3.)
    ys[I] = 0.5 * xs[I] * xs[I] - 3. * xs[I] + 4.5
    return ys

def dg_dx(xs_input, xms, del_x
    """
    Must synchronize with `g()`.
    """
    xs = np.atleast_1d((xs_input - xms) / del_x + 1.5) # 1.5 for re-center
    ys = np.zeros_like(xs)
    I = (xs > 0.) * (xs < 1.) 
    ys[I] = xs[I]
    I = (xs > 1.) * (xs < 2.)
    ys[I] = -2. * xs[I] + 3.
    I = (xs > 2.) * (xs < 3.)
    ys[I] = xs[I] - 3.
    return ys / del_x
