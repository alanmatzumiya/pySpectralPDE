from .tools import JNM, analysis, Cheby
from .solvers import set_simulation
from .grapher import graph_defaults


class setup_solver:
    """
    Solves the Burgers Stochastic equation on 1D using Spectral method
    based on the spectral decomposition of the Ornstein-Uhlenbeck semigroup
    associated to the Kolmogorov equation and compute the norm between two solutions
    with different initial conditions for a fixed point in real space.
    
    Solves the Fishers-KPP Stochastic equation on 1D using Spectral method
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

    def __init__(self, u0, params, equation="burgers"):

        self.get_data = set_simulation(u0, params, JNM).SimulaTX()
        self.u0_approx = Cheby.fit(u0, 0, 1, 3)
        self.plot = graph_defaults(params['x'], params['t'], self.get_data)
        self.stability = analysis(params['N']).distance



