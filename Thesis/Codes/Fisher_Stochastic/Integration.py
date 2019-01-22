from numpy import sin, pi, sqrt, exp, cos
import numpy as np
import itertools
from math import factorial
import scipy.integrate as integrate
from scipy.integrate import dblquad
from scipy import special

from IC import evalXSin
from JNM import Js

from numba import jit


def integr1(J, M, rulesX, rulesW, LRules, nu, u0):
    """
    Function to calculate first type of integrals.

    Parameters
    ----------
    J : array; shape([M, M])
        Array containing the grades of the polynomials
    M : int; max order polynomials
    rulesX : array; polynomials of Hermite evaluated
    rulesW : array; weights of polynomials of Hermite evaluated
    LRules : array; length of rulesX array
    nu : float; Diffusion coefficient
    u0 : callable(x); initial condition function

    Returns
    -------
    intg : float, value of first type of integral

    """


    intg = 0
    for k in range(0,LRules):
        prod = 1
        for i in range(0,M):
            if (J[i,0] > 0):
                x1 = evalXSin(nu, i + 1, u0)
                prod = prod * special.eval_hermitenorm(int(J[i,0]), rulesX[k])
        intg = intg + prod * rulesW[k]

    return intg

@jit
def integr2(N, Jm, Jn, M, rulesX, rulesW, LRules, nu, r, i):
    """
    Function to calculate second type of integrals.

    Parameters
    ----------
    N : int; max order polynomials
    Jm : int; column m of matrix J
    Jn : int; column m of matrix J
    M : int; number of ODEs
    rulesX : array; polynomials of Hermite evaluated
    rulesW : array; weights of polynomials of Hermite evaluated
    LRules : array; length of rulesX array
    nu : float; Diffusion coefficient
    r : int; column index of matrix J
    i : int; column index of matrix J

    Returns
    -------
    intg : array, size M
        Array containing the values of second type of integral

    """
    intg = np.zeros(M)
    for k in range(0, M):
        intg1 = 0

        if (Jn[k] > 0):
            sum2 = 0
            Jn1 = Js(N)[:,r]

            Jm1 = Js(N)[:,i]

            prod = 1.0
            for o in range(0, M):
                sumparc = 0

                for l in range(0, LRules):
                    x1 = rulesX[l]
                    factor1 = (1.0 / sqrt(factorial(int(Jn1[o])))) * (1.0 / sqrt(factorial(int(Jm1[o]))))
                    prod1 = special.eval_hermitenorm(int(Jn1[o]), x1) * special.eval_hermitenorm(int(Jm1[o]), x1) * factor1
                    sumparc = sumparc + prod1 * rulesW[l] / (sqrt(2.0 * nu) * pi * (o + 1))

                prod = prod * sumparc

            for l in range(0, LRules):
                x2 = rulesX[l]
                factor2 =(1.0 / sqrt(factorial(int(Jn[k]) - 1))) * (1.0 / sqrt(factorial(int(Jm[k]))))
                sum2 = sum2 + special.eval_hermitenorm(int(Jn[k]) - 1, x2) * special.eval_hermitenorm(int(Jm[k]), x2) * rulesW[l] * factor2

            intg1 = sum2 * prod
        intg[k] = intg1

    return intg


def integr3k(M):
    """
    Function to calculate third type of integrals.

    Parameters
    ----------
    M : int; size of individual timestep

    Returns
    -------
    int3 : array, size M
        Array containing the values of third type of integral

    """

    int3 = np.zeros(M)
    for k in range(1, M + 1):

        f = lambda x: sqrt(2.0) * sin(k * pi * x) * x
        integr = integrate.quad(f, 0, 1)  ### 1

        sum = integr[0]
        int3[k-1] = sum
    return int3


def integr4(M):
    """
    Function to simulate T-X.

    Parameters
    ----------
    M : int; size of individual timestep

    Returns
    -------
    int4 : array, size M
        Array containing the values of fourth type of integral

    """
    int4 = np.zeros(M)
    int3 = integr3k(M)
    for j in range(0, M):
        intM = np.zeros([M, M])
        for k in range(0, M):
            for l in range(0, M):
                f = lambda x: (2.0) * (sin((k + 1) * x) *  sin((l + 1) * pi * x)) * sin((j + 1) * pi * x)
                integr = integrate.quad(f, 0, 1)
                intM[l, k] = integr[0] * int3[l] * int3[k]
        int4[j] = int3[j] - sum(intM.sum(1))


    return int4
