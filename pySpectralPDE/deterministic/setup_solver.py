from numpy import pi, linspace, array, sin, round, arange
from .solvers import schemes
from .helpers import eval_aprox
from .grapher import graph_defaults
from .solvers import *
from .tools import *

class spectral_solver(object):

    def __init__(self, u0, params, equation="diffusion", scheme='galerkin', integrator_method='explicit'):

        self.N = N
        h = 2.0 * pi / N
        p = 2 * pi / (xR - xL)
        self.x = array(arange(- self.N / 2, self.N / 2), dtype=float) * h / p + xL
        self.t = linspace(t0, tmax, int(round(tmax / dt)) + 1, dtype=float)

        self.params = dict(nu=nu, N=N, xL=xL, xR=xR,
                           dt=dt, t0=t0, tmax=tmax,
                           p=p, x=self.x, tdata=self.t)
        self.data = schemes(self.params, u0, scheme, integrator_method).data
        self.scheme = scheme
        self.integrator_method = integrator_method
        self.plot = graph_defaults()

    def __call__(self, space, time):

        vt = self.data[self.t.index(time)]
        if self.scheme == 'galerkin':
            return eval_aprox().continuous_expansion(space, vt, self.N)
        else:
            return eval_aprox().discrete_expansion(space, vt, self.N)
