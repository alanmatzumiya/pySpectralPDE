from numpy import pi, linspace, array, sin, round
from .solvers import schemes
from .helpers import eval_aprox
from .grapher import graph_defaults


class deterministic_solver(object):

    def __init__(self, nu=0.0, N=16, xL=-1.0, xR=1.0, dt=0.1, t0=0.0, tmax=10.0, u0=sin,
                 integrator_method='explicit', scheme='galerkin', *args) -> object:

        self.N = N
        h = 2.0 * pi / N
        p = 2 * pi / (xR - xL)
        self.x = array([h * i for i in range(1, N + 1)], dtype=float) / p + xL
        self.t = linspace(t0, tmax, int(round(tmax / dt)) + 1, dtype=float)

        self.params = dict(nu=nu, N=N, xL=xL, xR=xR,
                           dt=dt, t0=t0, tmax=tmax,
                           p=p, x=x, tdata=self.t)
        self.data = schemes(self.params, u0, scheme, integrator_method)
        self.scheme = scheme
        self.integrator_method = integrator_method
        self.plot = graph_defaults()

    def __call__(self, space, time):

        vt = self.data[self.t.index(time)]
        if self.scheme == 'galerkin':
            return eval_aprox().continuous_expansion(space, vt, self.N)
        else:
            return eval_aprox().discrete_expansion(space, vt, self.N)