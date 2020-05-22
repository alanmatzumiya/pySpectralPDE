import numpy as np

from Simulation import SimulaTX, SimulaX
from IC import u0
from EigenF import EigeF, u02
from JNM import Js
from cheby import Cheby
from Convergence import conv_IC


def run_burgers():
    """
    Solves the Burgers Stochastic equation on 1D using Spectral method
    based on the spectral decomposition of the Ornstein-Uhlenbeck semigroup
    associated to the Kolmogorov equation and compute the norm between two solutions
    with different initial conditions for a fixed point in real space.

    Returns
    -------
    xSpace : array; discretized real space
    tim : array; discretized time
    simulation1 : array, shape(len(tim), len(xSpace))
        Array containing the solutions of partial equation
    simulation2 : array, shape(len(tim), len(xSpace))
        Array containing the solutions of partial equation with the IC approximated
    norms : array; norms between two solutions
    times : array; discretized time
    """
    # Diffusion coefficient
    nu = 0.01

    # Parameteres of the method
    N = 5
    Q = 200

    # Discretization
    xSpace = np.linspace(0,1, 2048)
    tim = np.linspace(0, 10, 1026)

    # Creating set J^{N;M}
    J = Js(N)
    M = len(J[:,1])

    # Hermite polynomials evaluation
    rule1 = np.polynomial.hermite_e.hermegauss(Q)
    rulesX = rule1[0][::-1]
    rulesW = rule1[1]
    LRules = len(rulesX)

    # Simulation Space-Time
    EigValRe, EigValIm, EigVecRe, EigVecIm, U_1 = EigeF(J, N, M, rulesX, rulesW, LRules, xSpace, nu, u0)
    H1 = SimulaX(J, M, xSpace, nu, u0)

    # Aproximation to u0
    aprox = Cheby.fit(u0, 0, 1, 3)
    H2 = SimulaX(J, M, xSpace, nu, aprox)
    U_2 = u02(J, M, rulesX, rulesW, LRules, xSpace, nu, aprox)

    simulation1 = SimulaTX(xSpace, tim, M, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, H1)
    simulation2 = SimulaTX(xSpace, tim, M, EigValRe, EigValIm, EigVecRe, EigVecIm, U_2, H2)

    # compute convergence
    norms, times = conv_IC(xSpace, tim, M, J, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, U_2)
      
    return xSpace, tim, simulation1, simulation2, norms, times


xSpace, tim, simulation1, simulation2, norms, times = run_burgers()


# save data
np.save('xSpace', xSpace)
np.save('times', tim)
np.save('data_1', simulation1)
np.save('data_2', simulation2)
np.save('norms', norms)

