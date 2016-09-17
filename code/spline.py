import numpy as np

class OnedUniformCubicSpline:

    def __init__(self, K, h, offset):
        self.K = K
        self.h = h
        self.offset = offset
        self.Ms = None

    def get_breakpoint(self, k):
        return offset + h * k

    def get_breakpoints(self):
        return offset + h * np.arange(K)

    def set_parameters(self, Ms):
        assert Ms.shape == (self.K, )
        self.Ms = Ms

    def get_ks(self, xs):
        """
        # bugs:
        - outside-of-range issues?
        """
        ks = np.floor((xs - self.offset) / self.h).astype(int)
        return ks

    def evaluate(self, xs):
        assert self.Ms is not None
        ys = np.zeros_like(xs)
        ks = self.get_ks(xs)
        xks = self.get_breakpoints()[ks]
        Mks = self.Ms[ks]
        Mkps = self.Ms[ks + 1]
        aks = (Mkps - Mks) / (6. * self.h)
        bks = Mks / 2.
        cks = 
        print(xs, ks, xks, Mks)

if __name__ == "__main__":
    import pylab as plt

    K = 32  # number of points
    h = 0.443 # spacing of points
    offset = 1.022 # location of k=0
    spline = OnedUniformCubicSpline(K, h, offset)
    spline.set_parameters(np.random.normal(size=K))

    xs = np.arange(0., 3., 0.2)
    spline.evaluate(xs)
