from numpy import sin, pi, sqrt, exp, cos
import numpy as np
from math import factorial
from scipy import special
from numba import jit

from IC import evalXSin

@jit
def SimulaX(J, M, xSpace, nu, u0):
    """
    Function to simulate X.

    Parameters
    ----------
    J : array; shape([M, M])
        Array containing the grades of the polynomials
    M : int; number of ODEs
    xSpace : array; discretized real space
    nu : array; Diffusion coefficient
    u0 : callable(x); initial condition function

    Returns
    -------
    H : array, shape(M, len(xSpace))
        Array containing hermite polynomials evaluated to simulate real space

    """

    Px = len(xSpace)
    H = np.zeros([M, Px])

    for k in range(0, Px):
        for j in range(1, M):
            prod = 1.0
            for i in range(0, M):
                if (J[i, j] > 0):
                    x1 = evalXSin(nu, i + 1, u0)
                    prod =  prod * special.eval_hermitenorm(int(J[i, j]), x1) * (1.0 / sqrt(factorial(J[i, j])))

            H[j, k] = prod
        x2 = xSpace[k]
        H[0, k] = special.eval_hermitenorm(2, x2) - special.eval_hermitenorm(1, x2) + 1

    return H

@jit
def SimulaT(tim, M, EigValRe, EigValIm, EigVecRe, EigVecIm, cons):
    """
    Function to simulate Time

    Parameters
    ----------
    tim : array; discretized time
    M : int; number of ODEs
    EigValRe : array, size M
        Array containing the real eigenvalues of matrix A
    EigValIm : array, size M
        Array containing the imaginary eigenvalues of matrix A
    EigVecRe : array, shape(M, M)
        Array containing the real eigenvectors of matrix A
    EigVecIm : array, shape(M, M)
        Array containing the imaginary eigenvectors of matrix A
    cons : array, shape(M, len(xSpace))
        Array containing the ordinary differential equations system constants

    Returns
    -------
    T : array, shape(len(tim), M)
        Array containing the solutions of each ODE for each zi on real space

    """

    Pt = len(tim)
    T = np.zeros([Pt,M])
    for k in range(0, Pt):  ### for each point t
        prod = np.zeros(M)

        for j in range(0, M):  ### for each multi-index n_j\in J^{N,M}
            sum = 0
            for i in range(0, M):  ### for each member inside the multi-index n_j=(\alpha_i,,,,)
                if (EigValIm[i] != 0):
                    if (i == 1):
                        sum = sum + cons[i] * exp(EigValRe[i] * tim[k] / pi**2) * (EigVecRe[j, i] * cos(EigValIm[i] * tim[k] / pi**2) - EigVecIm[j, i] * sin(EigValIm[i] * tim[k] / pi**2))
                if (i > 1):
                    if (EigValIm[i] != -EigValIm[i - 1]):
                        sum = sum + cons[i] * exp(EigValRe[i] * tim[k] / pi**2) * (EigVecRe[j, i] * cos(EigValIm[i] * tim[k] / pi**2) - EigVecIm[j, i] * sin(EigValIm[i] * tim[k] / pi**2))

                if (EigValIm[i] == -EigValIm[i - 1]):
                    sum = sum + cons[i] * exp(EigValRe[i] * tim[k] / pi**2) * (EigVecRe[j, i - 1] * sin(EigValIm[i - 1] * tim[k] / pi**2) + EigVecIm[j, i - 1] * cos(EigValIm[i - 1] * tim[k] / pi**2))

                if (EigValIm[i] == 0):
                    sum = sum + cons[i] * EigVecRe[j, i] * exp(EigValRe[i] * tim[k] / pi**2)
            prod[j] = sum
        T[k,:] = prod

    return T

@jit
def SimulaTX(xSpace, tim, M, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, H1):
    """
    Function to simulate Time-X.

    Parameters
    ----------
    xSpace : array; discretized real space
    tim : array; discretized time
    J : array; shape(M, M)
        Array containing the grades of the polynomials
    M : int; number of ODEs
    EigValRe : array, size M
        Array containing the real eigenvalues of matrix A
    EigValIm : array, size M
        Array containing the imaginary eigenvalues of matrix A
    EigVecRe : array, shape(M, M)
        Array containing the real eigenvectors of matrix A
    EigVecIm : array, shape(M, M)
        Array containing the imaginary eigenvectors of matrix A
    U_1 : array, shape(M, len(xSpace))
        Array containing the ordinary differential equations system constants
    nu : array; Diffusion coefficient
    H1 : array, shape(M, len(xSpace))
        Array containing hermite polynomials evaluated to simulate real space

    Returns
    -------
    abs(Tx) : array, shape(len(tim), len(xSpace))
        Array containing the solutions of partial equation

    """
    LX = len(xSpace)
    LT = len(tim)
    TX1 = np.zeros([LT, LX])
    TX = np.zeros([LT, LX])

    for x in range(0, LX):
        U1 = np.dot(np.linalg.inv(EigVecRe), U_1[:, x])
        Time1 = SimulaT(tim, M, EigValRe,EigValIm,EigVecRe,EigVecIm,U1)
        TX1 = np.dot(Time1, H1)
        TX[:, x] = TX1[:, x]

    return abs(TX)

@jit
def Un(tim, M, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, zi):
    """
    Function to calculate the solution of each ODE

    Parameters
    ----------
    tim : array; discretized time
    J : array; shape(M, M)
        Array containing the grades of the polynomials
    M : int; number of ODEs
    EigValRe : array, size M
        Array containing the real eigenvalues of matrix A
    EigValIm : array, size M
        Array containing the imaginary eigenvalues of matrix A
    EigVecRe : array, shape(M, M)
        Array containing the real eigenvectors of matrix A
    EigVecIm : array, shape(M, M)
        Array containing the imaginary eigenvectors of matrix A
    U_1 : array, shape(M, len(xSpace))
        Array containing the ordinary differential equations system constants
    zi : float; index of the point on xSpace

    Returns
    -------
    Utn : array, shape(M, len(tim))
    Array containing the solutions of each ODE in zi

    """

    Cons = np.dot(np.linalg.inv(EigVecRe), U_1[:, zi])
    Utn = SimulaT(tim, M, EigValRe,EigValIm,EigVecRe,EigVecIm, Cons)

    return Utn

