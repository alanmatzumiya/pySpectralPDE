from numpy import pi, sqrt, exp
import numpy as np
from scipy.integrate import dblquad
from scipy import special
from math import factorial
from Simulation import Un
from numba import jit


@jit
def measure(x):
    """
    This function calculates the Gaussian measure

    Parameters
    ----------
    x : array or float;
        array or float to compute the Gaussian measure

    Returns
    -------
    mu : float or array,
        Gaussian measure in x

    """

    mu = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x**2)
    return mu


def integ(J, M):
    """
    This function calculates the integral of the norm between two solutions
    with different initial conditions for a fixed point in real space

    Parameters
    ----------
    J : array; shape(M, M)
        Array containing the grades of the polynomials
    M : int; number of ODEs

    Returns
    -------
    Pn, Qn : array, shape(M, M)
        array containing the integrals of the Hermite polynomials

    """
    Pn = np.zeros([M, M])
    Qn = np.zeros([M, M])
    for j in range(0, M):
        for i in range(0, M):
            if J[i, j] > 0:
                Jij = int(J[i, j])
                P = lambda x, y: (abs(((-1)**Jij)*(special.eval_hermitenorm(Jij, x) - special.eval_hermitenorm(Jij, y)) / (factorial(Jij + 1)**2))** 2) * measure(x) * measure(y)
                Q = lambda x, y: (abs(((-1)**Jij)* special.eval_hermitenorm(Jij, y)/(factorial(Jij + 1)**2))** 2) * measure(x) * measure(y)
                integn1 = dblquad(P, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
                integn2 = dblquad(Q, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
                Pn[i, j] = integn1[0]
                Qn[i, j] = integn2[0]
    return Pn, Qn


@jit
def conv_IC(xSpace, tim, M, J, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, U_2):
    """
    This function calculates the norm between two solutions
    with different initial conditions for a fixed point in real space

    Parameters
    ----------
    xSpace : array; discretized real space
    tim : array; discretized time
    M : int; number of ODEs
    J : array; shape([M, M])
        Array containing the grades of the polynomials
    EigValRe : array, size M
        Array containing the real eigenvalues of matrix A
    EigValIm : array, size M
        Array containing the imaginary eigenvalues of matrix A
    EigVecRe : array, shape(M, M)
        Array containing the real eigenvectors of matrix A
    EigVecIm : array, shape(M, M)
        Array containing the imaginary eigenvectors of matrix A
    U_1 : array, shape(M, len(xSpace))
        Array containing the ordinary differential equations system constants to u0
    U_2 : array, shape(M, len(xSpace))
        Array containing the ordinary differential equations system constants to approximation

    Returns
    -------
    norms : array; norms between two solutions
    times : array; discretized time
    """

    Pn, Qn = integ(J, M)
    z = int(len(xSpace) / 50)
    Utn = Un(tim, M, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, z)
    Vtn = Un(tim, M, EigValRe, EigValIm, EigVecRe, EigVecIm, U_2, z)
    normas = []
    times = []
    for k in range(0,len(tim)):
        Ut = Utn[k,:]
        Vt = Vtn[k,:]
        sum = 0

        for j in range(0, M):
            for i in range(0, M):
                prod1 = 1.0
                prod2 = 1.0
                sum1 = Ut[J[i, j]]**2
                sum2 = (Ut[J[i, j]] - Vt[J[i, j]])**2
                for l in range(0, M):
                    if J[l, j] > 0:
                        prod1 = prod1 * Pn[l, j]
                        prod2 = prod2 * Qn[l, j]
                sum = sum + sum1 * prod1 + sum2 * prod2
        norma1 = sum
        normas.append(norma1)
        times.append(tim[k])

    return  normas, times