import numpy as np
from scipy.io.idl import readsav
from scipy.interpolate import interp1d
from wobble import read_harps, rv_model 
import h5py
c = 299792458. # m/s


def doppler(rv):
    beta = rv / c
    return np.sqrt((1. - beta) / (1. + beta))


if __name__ == "__main__":
        # load the data for quiet star HIP54287 (HARPS RMS 1.3 m/s)
    # BUGS: directories and dependencies will only work on Megan's computer...
   
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    s = readsav(data_dir+'HIP30037_result.dat')
    Star = rv_model.RV_Model()

    # grab a tellurics section from s1d files
    true_rvs = ( -s.berv + s.rv) * 1.e3  # m/s
    #true_rvs = (s.rv - 54.9) * 1.e3 
    drift = s.drift # m/s
    dx = 0.01 # A
    #xs = np.arange(4998.0, 5002.0, dx)
    xs = np.arange(3782.0, 6910.0, dx) # tellurics region
    N = len(s.files)  # number of epochs
    M = len(xs)
    data = np.empty((N, M))
    ivars = np.empty_like(data)
    for n,(f,b,snr) in enumerate(zip(s.files, s.berv, s.snr)):
            # read in the spectrum
            spec_file = str.replace(f, 'ccf_G2', 's1d')
            wave, spec = read_harps.read_spec(spec_file)
            # re-introduce barycentric velocity
            wave *= doppler(b*1.e3)
            # remove systemic RV shift so we're looking at the same lines as example
            #wave *= doppler(54.9 * 1.e3)
            # save the relevant bit
            f = interp1d(wave, spec)
            data[n,:] = f(xs)
            ivars[n,:] = snr**2
            print wave[0], wave[-1]
            
    h = h5py.File('../data/hip30037.hdf5', 'w')
    dset = h.create_dataset('data', data=data)
    dset = h.create_dataset('ivars', data=ivars)
    dset = h.create_dataset('xs', data=xs)
    dset = h.create_dataset('true_rvs', data=true_rvs)
    dset = h.create_dataset('date', data=s.date)
    dset = h.create_dataset('berv', data=s.berv)
    
    h.close()
    
    
    
            
    