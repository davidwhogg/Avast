"""
This file is part of the Avast project.
Copyright 2016 David W. Hogg (NYU).
"""
import numpy as np
c = 3.e8 # m / s

class OneDGaussianModel():

    def __init__(self, K, h, offset, baseline=0.):
        assert K > 1
        assert h > 0.
        self.K = K
        self.h = h
        self.means = np.arange(K) * h + offset
        self.var = (1.5 * h) ** 2 # well sampled
        self.offset = offset
        self.baseline = baseline
        self.amps = None

    def _OneDGaussians(self, xs):
        return self.amps[None,:] * np.exp(-0.5 * (xs[:,None] - self.means[None,:]) ** 2 / self.var)

    def _d_OneDGaussians_dx(self, xs):
        return self._OneDGaussians(xs) * (self.means[None, :] - xs[:,None]) / self.var

    def set_amplitudes(self, amps):
        assert amps.shape == (self.K, )
        self.amps = amps

    def evaluate(self, xs):
        N = len(xs)
        assert xs.shape == (N, )
        return np.sum(self._OneDGaussians(xs), axis=1) + self.baseline

    def evaluate_derivative(self, xs):
        N = len(xs)
        assert xs.shape == (N, )
        return np.sum(self._d_OneDGaussians_dx(xs), axis=1)

    def evaluate_at_redshift(self, xs, rv):
        restxs = redshift_wavelengths(xs, -rv)
        return self.evaluate(restxs)

    def evaluate_at_redshift_linearized(self, xs, rv):
        return self.evaluate(xs) - xs * self.evaluate_derivative(xs) * (rv / c)

def redshift_wavelengths(waves, v):
    beta = v / c
    return waves * np.sqrt((1. + beta) / (1. - beta))

def make_fake_data():
    np.random.seed(17)
    N = 50 # number of spectra
    K = 1000  # number of control points
    h = 1. # spacing of control points (Ang)
    offset = 4500. # location of k=0 control point (Ang)
    truth = OneDGaussianModel(K, h, offset, 1.0)
    truth.set_amplitudes(-0.5 * np.random.uniform(size=K) ** 32.) # made up
    times = (np.random.uniform(size=N) - 0.5) * 1400. # four+ years
    rvs = 2200. * np.sin(2. * np.pi * times / 55.81) # m / s
    waves = 1. * truth.means # copy
    data = np.zeros((N, len(waves)))
    for n in range(N):
        restwaves = redshift_wavelengths(waves, -rvs[n])
        data[n, :] = truth.evaluate(restwaves)
    data += 0.01 * np.random.normal(size=data.shape) # s/n = 100
    return data, waves

if __name__ == "__main__":
    import pylab as plt
    np.random.seed(42)

    K = 2 ** 10  # number of points
    h = 0.443 # spacing of points
    offset = 4999. # location of k=0
    model = OneDGaussianModel(K, h, offset)
    model.set_amplitudes(np.random.normal(size=K))

    xs = np.arange(5000, 5010., 0.01)
    plt.clf()
    plt.plot(xs, model.evaluate(xs), "k-")
    plt.plot(xs, model.evaluate_derivative(xs), "r-")
    plt.axhline(0., color="r", alpha=0.25)
    plt.savefig("test.png")

    model.set_amplitudes(np.ones(K))
    plt.clf()
    plt.plot(xs, model.evaluate(xs), "k-")
    plt.plot(xs, model.evaluate_derivative(xs), "r-")
    plt.axhline(0., color="r", alpha=0.25)
    plt.savefig("test2.png")

    model.set_amplitudes(-0.5 * np.random.uniform(size=K) ** 32.) # made up
    model.baseline = 1.
    plt.clf()
    plt.plot(xs, model.evaluate(xs), "k-")
    plt.plot(xs, model.evaluate_derivative(xs), "r-")
    delta = 0.01 * np.sqrt(model.var)
    numerical_derivative = (model.evaluate(xs + delta) - model.evaluate(xs)) / delta
    plt.plot(xs, numerical_derivative, "g--", alpha=0.5)
    plt.axhline(0., color="r", alpha=0.25)
    plt.savefig("test3.png")

    plt.clf()
    plt.plot(xs, model.evaluate(xs), "k-")
    rv = 1.e4 # m/s
    plt.plot(xs, model.evaluate_at_redshift(xs, rv), "r-")
    plt.plot(xs, model.evaluate_at_redshift_linearized(xs, rv), "g--", alpha=0.5)
    plt.savefig("test4.png")

if False:    
    data, waves = make_fake_data()
    N, M = data.shape
    n = 0
    plt.clf()
    plt.step(waves, data[n, :], color="k", where="mid")
    plt.xlabel("wavelength (Angstrom)")
    plt.ylabel("normalized flux")
    plt.savefig("data00.png")

