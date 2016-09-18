"""
This file is part of the Avast project.
Copyright 2016 David W. Hogg (NYU).
"""
import numpy as np

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

def make_fake_data():
    np.random.seed(17)
    N = 50 # number of spectra
    K = 1000  # number of control points
    h = 1. # spacing of control points (Ang)
    offset = 4500. # location of k=0 control point (Ang)
    c = 3.e8 # m / s
    truth = OneDGaussianModel(K, h, offset, 1.0)
    truth.set_amplitudes(-0.5 * np.random.uniform(size=K) ** 32.) # made up
    times = (np.random.uniform(size=N) - 0.5) * 1400. # four+ years
    rvs = 2.2 * np.sin(2. * np.pi * times / 55.81) # m / s
    waves = 1. * truth.means # copy
    data = np.zeros((N, len(waves)))
    for n in range(N):
        data[n, :] = truth.evaluate(waves) - (waves / c) * truth.evaluate_derivative(waves) * rvs[n]
    data += 0.01 * np.random.normal(size=data.shape) # s/n = 100
    return data, waves

if __name__ == "__main__":
    import pylab as plt

    data, waves = make_fake_data()
    N, M = data.shape
    plt.clf()
    n = 0
    plt.step(waves, data[n, :], color="k", where="mid")
    plt.xlabel("wavelength (Angstrom)")
    plt.ylabel("normalized flux")
    plt.savefig("data00.png")

if False:
    np.random.seed(42)

    K = 2 ** 10  # number of points
    h = 0.443 # spacing of points
    offset = 1.022 # location of k=0
    model = OneDGaussianModel(K, h, offset)
    model.set_amplitudes(np.random.normal(size=K))

    xs = np.arange(-5., 100., 0.01)
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
    plt.axhline(0., color="r", alpha=0.25)
    plt.savefig("test3.png")
