"""
This file is part of the Avast project.
Copyright 2016 Megan Bedell (Chicago) and David W. Hogg (NYU).
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io.idl import readsav
from harps_tools import read_harps
import cPickle as pickle
c = 2.99792458e8   # m/s

if __name__ == "__main__":
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    s = readsav(data_dir+'HIP22263_result.dat')
    wave_lo = 6244.0
    wave_hi = 6249.0
        
    print "Reading files..."
    filelist = [str.replace(f, 'ccf_G2', 's1d') for f in s.files]
    nfile = len(filelist)
    data = []
    for e,fn in enumerate(filelist):
        w, f = read_harps.read_spec(fn)
        w = w[f > 0.]
        f = f[f > 0.]
        data.append((np.log(w),f,np.sqrt(f)))
    pickle.dump(data, open( "data.p", "wb" ), -1)  