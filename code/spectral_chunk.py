"""
This file is part of the Avast project.
Copyright 2016 Megan Bedell (Chicago) and David W. Hogg (NYU).
"""
import triangle_basis
from triangle_basis import xi_scale, scale_scale
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize
import cPickle as pickle
c = 2.99792458e8   # m/s

def read_spec(spec_file):
    # read a HARPS 1D spectrum file from the ESO pipeline
    sp = fits.open(spec_file)
    header = sp[0].header
    n_wave = header['NAXIS1']
    crval1 = header['CRVAL1']
    cdelt1 = header['CDELT1']
    index = np.arange(n_wave, dtype=np.float64)
    wave = crval1 + index*cdelt1
    spec = sp[0].data
    obj = header['OBJECT']
    bjd = header['ESO DRS BJD']
    return wave, spec, obj, bjd

def solve_chunk(pars0, xs, ys, yerrs, xms, del_x):
    # given starting guess of pars0, solve for best-fit pars in a chunk.
    # returns OptimizeResult object.
    fa = (xs, ys, yerrs, xms, del_x)
    res = minimize(triangle_basis.obj_function, pars0, args=fa, 
        method='BFGS', jac=triangle_basis.obj_deriv)
    return res

def read_chunk(spec_file, start_wave, save_dir):
    # for a given 1D spectrum file at a given chunk, return the saved solution.
    sp = fits.open(spec_file)
    header = sp[0].header
    obj = header['OBJECT']
    bjd = header['ESO DRS BJD']
    save_name = save_dir+'{0}_d{1}_w{2:4}.p'.format(obj,str(bjd),start_wave)
    xs, ys, yerrs, res, obj, bjd = pickle.load(open(save_name, "rb"))
    return xs, ys, yerrs, res, obj, bjd
    
def solve_spec(spec_file, number, save_dir):
    # reads in a HARPS s1d FITS file and solves it chunk-by-chunk.
    # output is saved as a series of pickle files under path save_dir.
    print "Solving spectrum {0}...".format(spec_file)
    wave, spec, obj, bjd = read_spec(spec_file)
    
    del_x = 1.3e-5/2.0
    
    #for mid_ind in range(500, len(wave)-500, 500):
    for mid_ind in range(277320, 277320+2000, 500): # for testing
        pad = 100 # number of points to overlap with neighboring chunk
        start_ind, stop_ind = mid_ind - 250 - pad, mid_ind + 250 + pad
        print "Solving chunk from {0:4.1f} A to {1:4.1f} A...".format(wave[start_ind], wave[stop_ind])
        start_wave = wave[start_ind]
        xs = [np.log(wave[start_ind:stop_ind])]
        ys = [spec[start_ind:stop_ind]]
        yerrs = np.sqrt(ys) # TODO: fix this
        
        xms = np.arange(np.min(xs) - 0.5 * del_x, np.max(xs) + 0.99 * del_x, del_x)
        n_ms = len(xms)
        ams0 = np.ones(n_ms)
        scales0 = [np.median(s) * scale_scale for s in ys]
        xis0 = np.zeros(1) * xi_scale
        pars0 = np.append(ams0, np.append(scales0, xis0))
        
        res = solve_chunk(pars0, xs, ys, yerrs, xms, del_x)
        save_obj = (xs, ys, yerrs, res, obj, bjd)
        save_name = save_dir+'{0}_e{1}_w{2:04}.p'.format(obj,number,int(start_wave))
        print "Saving chunk as {0}...".format(save_name)
        pickle.dump(save_obj, open(save_name, "wb"))
        

if __name__ == "__main__":
    data_dir = '../data/chunk_test/'    
    filelist = glob.glob(data_dir+'*_s1d_A.fits')
    for i,f in enumerate(filelist): 
        if i>2:
            continue  # for testing
        solve_spec(filelist[0], i, data_dir)
    