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
    #obj = header['OBJECT']
    #bjd = header['ESO DRS BJD']
    return wave, spec

def solve_chunk(pars0, xs, ys, yerrs, xms, del_x):
    # given starting guess of pars0, solve for best-fit pars in a chunk.
    # returns OptimizeResult object.
    fa = (xs, ys, yerrs, xms, del_x)
    res = minimize(triangle_basis.obj_function, pars0, args=fa, 
        method='BFGS', jac=triangle_basis.obj_deriv)
    return res


def solve_all(all_xs, all_ys, save_dir):
    # takes a list of HARPS spectra and solves them chunk-by-chunk.
    # output is saved as a series of pickle files under path save_dir.
    
    del_x = 1.3e-5/2.0
    
    #for mid_ind in range(500, len(all_xs[0])-500, 500):
    for mid_ind in range(277320, 277320+2000, 500): # for testing
        pad = 100 # number of points to overlap with neighboring chunk
        start_ind, stop_ind = mid_ind - 250 - pad, mid_ind + 250 + pad
        print "Solving chunk from {0:4.1f} A to {1:4.1f} A...".format(all_xs[0,start_ind], all_xs[0,stop_ind])
        start_wave = np.exp(all_xs[0,start_ind])
        xs = all_xs[:,start_ind:stop_ind]
        ys = all_ys[:,start_ind:stop_ind]
        yerrs = np.sqrt(ys) # TODO: fix this
        
        xms = np.arange(np.min(xs) - 0.5 * del_x, np.max(xs) + 0.99 * del_x, del_x)
        n_ms = len(xms)
        ams0 = np.ones(n_ms)
        scales0 = [np.median(s) * scale_scale for s in ys]
        xis0 = np.zeros(len(xs)) * xi_scale
        pars0 = np.append(ams0, np.append(scales0, xis0))
        
        res = solve_chunk(pars0, xs, ys, yerrs, xms, del_x)
        save_obj = (xs, ys, yerrs, xms, res)
        save_name = save_dir+'{0}_w{2:04}.p'.format('test',int(start_wave)) # TODO: add metadata
        print "Saving chunk as {0}...".format(save_name)
        pickle.dump(save_obj, open(save_name, "wb"))
        

if __name__ == "__main__":
    data_dir = '../data/chunk_test/'    
    filelist = glob.glob(data_dir+'*_s1d_A.fits')
    #nfile = len(filelist)
    nfile = 3 # for testing
    all_xs = None
  
    for e,fn in enumerate(filelist):
        wave, spec = read_spec(fn)
        if all_xs is None:
            nwave = len(wave)
            all_xs, all_ys = np.zeros([nfile,nwave]), np.zeros([nfile,nwave])
        assert len(wave) == nwave # THIS DOESN'T WORK - not all spectra have the same nwave!
        if e>2:
            continue  # for testing
        all_xs[e] = np.log(wave)
        all_ys[e] = spec
        
    solve_all(xs, ys, data_dir)
        
    for p in glob.glob(data_dir+'*.p'):
        xs, ys, yerrs, xms, res = pickle.load(open(p, "rb"))
        pars = res['x']
        n_epoch = len(xs)
        n_ms = len(xms)
        del_x = xms[1] - xms[0]
        ams, scales, xis = triangle_basis.unpack_pars(pars, n_ms, n_epoch)
        calcs = np.zeros((n_epoch, len(xs[e])))
        for e in range(n_epoch):
            xprimes = xs[e] + xis[e]
            calc = triangle_basis.f(xprimes, xms, del_x, ams * scales[e])
            x_plot = np.linspace(xprimes[0],xprimes[-1],num=5000)
            calc_plot = triangle_basis.f(x_plot+xis[e], xms, del_x, ams * scales[e])
            resid = triangle_basis.resid_function(pars, xs, ys, yerrs, xms, del_x)
            n_x = len(xs[e])
            resid = resid[e*n_x:(e+1)*n_x] * yerrs[e]
            triangle_basis.save_plot(xs[e], ys[e], calc, resid, x_plot, calc_plot, 
                'fig/wave{0:04}_epoch{1}.pdf'.format(int(np.exp(xs[e,0])),str(e)), e)
            calcs[e] = calc
        

    