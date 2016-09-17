"""
This file is part of the Avast project.
Copyright 2016 David W. Hogg (NYU).
"""
import numpy as np

class OneDGaussianModel():

    def __init__(self, K, h, offset):
        assert K > 1
        assert h > 0.
        self.K = K
        self.h = h
        self.means = np.arange(K) * h + offset
        self.var = (1.5 * h) ** 2 # well sampled
        self.offset = offset
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
        return np.sum(self._OneDGaussians(xs), axis=1)

    def evaluate_derivative(self, xs):
        N = len(xs)
        assert xs.shape == (N, )
        return np.sum(self._d_OneDGaussians_dx(xs), axis=1)

if __name__ == "__main__":
    import pylab as plt

    K = 2 ** 16  # number of points
    h = 0.443 # spacing of points
    offset = 1.022 # location of k=0
    model = OneDGaussianModel(K, h, offset)
    model.set_amplitudes(np.random.normal(size=K))

    xs = np.arange(-5., 20., 0.01)
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
