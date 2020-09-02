import numpy as np
from .tools import JNM, analysis, Cheby
from .solvers import set_simulation
from .grapher import graph_defaults

class stochastic_solver(object):
    """
    Solves the Burgers Stochastic equation on 1D using Spectral method
    based on the spectral decomposition of the Ornstein-Uhlenbeck semigroup
    associated to the Kolmogorov equation and compute the norm between two solutions
    with different initial conditions for a fixed point in real space.

     Parameters
    ----------
    nu : array; Diffusion coefficient
    N : int; order max of the polynomials
    xSpace : array; discretized real space
    tim : array; discretized time

    """

    def __init__(self, nu=0.01, N=5, Q=200, xSpace=np.linspace(0, 1, 256),
                 tim=np.linspace(0, 10, 128)) -> object:
        # Creating set J^{N;M}
        N = N
        J = JNM(N).Js()
        M = len(J[:, 1])

        # Hermite polynomials evaluation
        rule1 = np.polynomial.hermite_e.hermegauss(Q)
        rulesX = rule1[0][::-1]
        rulesW = rule1[1]
        LRules = len(rulesX)

        self.data = []
        self.norm = []

        self.params = dict(
            nu=nu, N=N, Q=Q, xSpace=xSpace, tim=tim, J=J, M=M,
            rulesX=rulesX, rulesW=rulesW, LRules=LRules
        )

        self.plot = graph_defaults()

    def built_simulation(self, u0):

        self.data = set_simulation(u0, self.params).SimulaTX()

    def get_stability(self, u0):

        aprox_IC = Cheby.fit(u0, 0, 1, 3)
        u = set_simulation(u0, self.params).Un()
        v = set_simulation(u0, self.params).Un()
        analysis(**self.params).distance(u, v)

