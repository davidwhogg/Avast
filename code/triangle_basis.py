"""
This file is part of the Avast project.
Copyright 2016 Megan Bedell (Chicago) and David W. Hogg (NYU).
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import leastsq, fmin_bfgs, fmin_cg
from scipy.linalg import svd
from scipy.io.idl import readsav
c = 2.99792458e8   # m/s
xi_scale = 1.e7
scale_scale = 1.e-4

def unpack_pars(pars, n_ms, n_epoch):
    # unpack parameters
    ams = pars[0:n_ms]
    scales = pars[n_ms:n_ms+n_epoch] / scale_scale
    xis = pars[n_ms+n_epoch:] / xi_scale
    return ams, scales, xis

def xi_to_v(xi):
    # translate ln(wavelength) Doppler shift to a velocity in m/s
    r = np.exp(2.*xi) # r = (c+v)/(c-v)
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

def resid_function(pars, xs, ys, yerrs, xms, del_x):
    """
    function to minimize

    ## bugs:
    - needs proper comment header.
    - array indexing is very brittle.
    """
    n_epoch = len(xs)
    n_ms = len(xms)
    n_x = len(xs[0])  # assumes all xs are the same length
    ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
    resid = np.zeros((n_epoch*n_x + n_ms + n_epoch))
    for j in range(n_epoch):
        xprimes = xs[j] + xis[j]
        calc = scales[j] * f(xprimes, xms, del_x, ams)
        resid[j*n_x:(j+1)*n_x] = (ys[j] - calc) / yerrs[j]
    resid[-n_epoch-n_ms:-n_epoch] = (ams - 1.)/1.0  # append penalty terms of a MAGIC NUMBER
    resid[-n_epoch:] = (xis - 0.)/3.e-6  # MAGIC NUMBER (1 km/s-ish)
    return resid.flatten()

def deriv_matrix(pars, xs, ys, yerrs, xms, del_x):
    """
    derivatives of resid_function() wrt pars

    ## bugs:
    - penalty terms are arbitrary & should be linked to resid_function
    """
    n_epoch = len(xs)
    n_ms = len(xms)
    n_x = len(xs[0])  # assumes all xs are the same length
    ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
    deriv_matrix = np.zeros((n_epoch*n_x + n_ms + n_epoch, len(pars)))
    for j in range(n_epoch):
        xprimes = xs[j] + xis[j]
        dy_dams = scales[j] * g(xprimes[:,None], xms[None,:], del_x)
        deriv_matrix[j*n_x:(j+1)*n_x,:n_ms] = - dy_dams / (yerrs[j])[:,None]
        dy_dsj = f(xprimes, xms, del_x, ams)
        deriv_matrix[j*n_x:(j+1)*n_x,n_ms+j] = - dy_dsj / yerrs[j] / scale_scale
        dy_dxij = scales[j] * np.sum(ams[None,:] * dg_dx(xprimes[:,None], 
            xms[None,:], del_x), axis=1)
        deriv_matrix[j*n_x:(j+1)*n_x,-n_epoch+j] = - dy_dxij / yerrs[j] / xi_scale
    deriv_matrix[-n_epoch-n_ms:-n_epoch,:n_ms] = np.eye(n_ms)/1.0  # append penalty terms of a MAGIC NUMBER
    deriv_matrix[-n_epoch:,-n_epoch:] = np.eye(n_epoch)/3.e-6 / xi_scale  # MAGIC NUMBER (1 km/s-ish)
    return deriv_matrix
    
def objective(pars, xs, ys, yerrs, xms, del_x):
   # scalar objective function
   resid = resid_function(pars, xs, ys, yerrs, xms, del_x)
   return np.dot(resid, resid)
   
def obj_deriv(pars, xs, ys, yerrs, xms, del_x):
    # derivative of objective function
    resid = resid_function(pars, xs, ys, yerrs, xms, del_x)
    matrix = deriv_matrix(pars, xs, ys, yerrs, xms, del_x)
    return 2.0 * np.dot(resid, matrix)

def min_v(pars, i, xs, ys, yerrs, xms, del_x):
    # do a simple minimization of just one xi parameter
    # i : epoch # to minimize, 0-2
    tmp_pars = np.copy(pars)    
    obj = []  # objective function
    xi0 = []
    for xi in np.linspace(xis[i]-1.,xis[i]+1.,100):
        tmp_pars[-n_epoch+i] = xi
        resids = resid_function(tmp_pars, xs, ys, yerrs, xms, del_x)
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
    
def save_plot(xs, obs, calc, resid, x_plot, calc_plot, save_name, i):
    xs = np.e**xs
    x_plot = np.e**x_plot
    fig = plt.figure()

    ax1 = fig.add_subplot(2,1,1)
    ax1.step(xs,obs, color='black', label='Observed')
    ax1.plot(x_plot,calc_plot, color='red', label='Calculated')
    ax1.set_ylabel('Flux')
    #ax1.legend()
    ax1.set_title('Epoch {0}'.format(i))
    ax1.set_xticklabels( () )
    ax2 = fig.add_subplot(2,1,2)
    ax2.step(xs,obs - calc, color='black')
    ax2.step(xs,resid, color='red')
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
    ax1.set_xlim([x_plot.min(),x_plot.max()])
    ax2.set_xlim([x_plot.min(),x_plot.max()])
    
    plt.savefig(save_name)

if __name__ == "__main__":
    data_dir = '../data/halpha_quiet/'
    
    print "Reading files..."
    filelist = glob.glob(data_dir+'*.txt')
    nfile = len(filelist)
    xs = None
    for e,fn in enumerate(filelist):
        w, s = np.loadtxt(fn, unpack=True)
        if xs is None:
            nwave = len(w)
            xs, ys = np.zeros([nfile,nwave]), np.zeros([nfile,nwave])
        assert len(w) == nwave
        xs[e] = np.log(w)
        ys[e] = s

    print "Got data!"
    yerrs = np.sqrt(ys)  # assumes Poisson noise and a gain of 1.0
    del_x = 1.3e-5/2.0
    xms = np.arange(np.min(xs) - 0.5 * del_x, np.max(xs) + 0.99 * del_x, del_x)
    
    # initial fit to ams & scales:
    fa = (xs, ys, yerrs, xms, del_x)
    n_epoch = len(xs)
    n_ms = len(xms)
    ams0 = np.ones(n_ms)
    scales0 = [np.median(s) * scale_scale for s in ys]
    xis0 = np.zeros(n_epoch) * xi_scale
    #xis0 = np.random.normal(size=n_epoch)/1.e7 # ~10 m/s level
    pars0 = np.append(ams0, np.append(scales0, xis0))
    ftol = 1.49012e-08  # default is 1.49012e-08 
    
    '''''
    soln = leastsq(resid_function, pars0, args=fa, ftol=ftol, full_output=True)
    pars_nodf = soln[0]   
    ams, scales, xis = unpack_pars(pars_nodf, n_ms, n_epoch)
    for e in range(n_epoch):
        xprimes = xs[e] + xis[e]
        calc = f(xprimes, xms, del_x, ams) * scales[e]
        x_plot = np.linspace(xprimes[0],xprimes[-1],num=5000)
        calc_plot = f(x_plot, xms, del_x, ams * scales[e])
        save_plot(xprimes, ys[e], calc, x_plot, calc_plot, 'fig/epoch'+str(e)+'_nodf.pdf')
    '''
    
    #soln = leastsq(resid_function, pars0, args=fa, Dfun=deriv_matrix, 
    #        col_deriv=False, ftol=ftol, full_output=True)  
    
    print "Optimizing...."
    
    gtol = 1.e-9
    soln = fmin_bfgs(objective, pars0, args=fa, fprime=obj_deriv, full_output=True, gtol=gtol)  
    print "Solution achieved!"

    # look at the fit:
    pars = soln[0]
    #pars = soln
    ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
    #resids = resid_function(pars, xs, ys, xms, del_x)
    print "Initial optimization of all parameters:"
    #print "Objective function value: {0}".format(np.dot(resids,resids))
    #print "nfev: {0}".format(soln[2]['nfev'])
    #print "mesg: {0}".format(soln[3])
    #print "ier: {0}".format(soln[4])
    vs = xi_to_v(xis)
    print "Velocities:", vs
    

    calcs = np.zeros((n_epoch, len(xs[e])))
    for e in range(n_epoch):
        xprimes = xs[e] + xis[e]
        calc = f(xprimes, xms, del_x, ams * scales[e])
        x_plot = np.linspace(xprimes[0],xprimes[-1],num=5000)
        calc_plot = f(x_plot+xis[e], xms, del_x, ams * scales[e])
        resid = resid_function(pars, xs, ys, yerrs, xms, del_x)
        n_x = len(xs[e])
        resid = resid[e*n_x:(e+1)*n_x] * yerrs[e]
        #save_plot(xs[e], ys[e], calc, resid, x_plot, calc_plot, 'fig/epoch'+str(e)+'.pdf', e)
        calcs[e] = calc
        
    scaled_resids = (ys - calcs) / scales[:,None]
    u, s, v = svd(scaled_resids, full_matrices=False)
    u.shape, s.shape, v.shape
    
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    pipeline = readsav(data_dir+'HIP54287_result.dat')
    plt.scatter(pipeline.rv, u[:,0])
    
    
if False:
    # plotting objective function with various parameters:
    tmp_pars = np.copy(pars)
    obj = []  # objective function
    a20 = []
    for a in np.linspace(ams[20]-100.,ams[20]+100.,100):
        tmp_pars[20] = a
        resids = resid_function(tmp_pars, xs, ys, yerrs, xms, del_x)
        obj = np.append(obj, np.dot(resids,resids))
        a20 = np.append(a20,a)
    plt.clf()
    plt.plot(a20,obj)
    plt.axvline(ams[20], linestyle='solid')
    plt.xlabel(r'a$_{20}$')
    plt.ylabel('objective function')
    plt.savefig('objectivefn_a20.png')
    plt.clf()
    
    tmp_pars = np.copy(pars)
    obj = []  # objective function
    scale0 = []
    for s in np.linspace(scales[0]*0.95,scales[0]*1.05,100):
        tmp_pars[n_ms] = s
        resids = resid_function(tmp_pars, xs, ys, yerrs, xms, del_x)
        obj = np.append(obj, np.dot(resids,resids))
        scale0 = np.append(scale0,s)
    plt.clf()
    plt.plot(scale0,obj)
    plt.axvline(scales[0], linestyle='solid')
    plt.xlabel(r'scale$_{0}$')
    plt.ylabel('objective function')
    plt.savefig('objectivefn_scale0.png')
    plt.clf()
    
    tmp_pars = np.copy(pars)    
    obj = []  # objective function
    xi0 = []
    for xi in np.linspace(xis[0]*0.95,xis[0]*1.05,100):
        tmp_pars[n_ms+n_epoch] = xi
        resids = resid_function(tmp_pars, xs, ys, yerrs, xms, del_x)
        obj = np.append(obj, np.dot(resids,resids))
        xi0 = np.append(xi0,xi)
    plt.clf()
    plt.plot(xi0,obj)
    plt.axvline(xis[0], linestyle='solid')
    plt.xlabel(r'$\xi_{0}$')
    plt.ylabel('objective function')
    plt.savefig('objectivefn_xi0.png')
    plt.clf()


    '''''
    # optimize one epoch at a time:
    for i in range(n_epoch):
        pars = min_v(pars, i, xs, ys, xms, del_x)
        resids = resid_function(pars, xs, ys, xms, del_x)
        ams, scales, xis = unpack_pars(pars, n_ms, n_epoch)
        print "Optimization of velocity at epoch {0}:".format(i)
        print "Objective function value: {0}".format(np.dot(resids,resids))
        #vs = xi_to_v(xis)
        #print "Velocities:", vs
    '''
