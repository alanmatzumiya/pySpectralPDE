from numpy import sin, pi, sqrt, exp, cos
import numpy as np
import itertools
from math import factorial
import scipy.integrate as integrate
from scipy.integrate import dblquad
from scipy import special


def J_NM(N, M, N1):
    """
    function to construct the J^{N;M} polynomial products by column.

    Parameters
    ----------
    N : int; order max of the polynomials
    M : int; number of ODEs
    N1 : int; order of the polynomials

    Returns
    -------
    J : array, shape(len(M), len(M))
        Array containing the grades of the polynomials.

    """

    J = np.zeros([M, M], dtype=int)
    L1 = 5

    for k in range(1, N1):
        h = list(itertools.combinations(range(3, N + 1), k))
        L = len(h)
        L2 = L1 + L - 1
        J[0, L1-1:L2] = 1
        J[1, L1-1:L2] = 2
        for j in range(0, L):
            for i in range(0, len(h[j])):
                J[2  + i , L1 + j - 1] = h[j][i]
        L1 = L1 + L

    J[0, 0] = 2
    J[0, 1] = 1
    J[1, 1] = 2

    J[0, 2] = 1
    J[1, 2] = 2
    J[2, 2] = 1

    J[0, 3] = 1
    J[1, 3] = 2
    J[2, 3] = 2
    return J


def Js(N):
    """
    Function to compute set J

    Parameters
    ----------
    N : int; max order of polynomial

    Returns
    -------
    J1 : array, shape(len(M), len(M))
        Array containing the grade of each Hermite polynomial

    """
    M = 0
    if (N == 4):
        M = 7
        N1 = 3
        J1 = J_NM(N, M, N1)

    if (N == 5):
        M = 11
        N1 = 4
        J1 = J_NM(N, M, N1)

    if (N == 6):
        M = 18
        N1 = 4
        J2 = J_NM(N, M, N1)
        SeqJ = np.array([1, 3] + list(range(5,7)) + [8] +  list(range(11,14)) + [15, 17]) - 1  #### This is with the numbers (N=6, M=18, N1=4),
        M = 10
        J1 = J2[0:M, SeqJ]

    if (N == 7):
        M = 29
        N1 = 4
        J2 = J_NM(N, M, N1)
        SeqJ = np.array(list(range(1, 10)) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29]) - 1  #### This is with the numbers (N=7, M=29, N1=4),
        M = 20
        J1 = J2[0:M, SeqJ]

    if (N == 8):
        M = 60
        N1 = 5
        J2 = J_NM(N, M, N1)
        SeqJ = np.array(list(range(1,10)) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29, 31, 34, 35, 38, 39, 44, 48, 51, 55, 59]) - 1  #### This is with the numbers (N=7, M=29, N1=4),
        M = 30
        J1 = J2[0:M, SeqJ]

    if (N == 9):
        M = 150
        N1 = 5
        J2 = J_NM(N, M, N1)
        SeqJ = np.array(list(range(1, 10)) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29, 31, 34, 35, 38, 39, 44, 48, 51, 55, 59, 63, 67, 70, 75, 81, 85, 89, 92, 95, 99]) - 1  #### This is with the numbers (N=7, M=29, N1=4),
        M = 40
        J1 = J2[0:M, SeqJ]

    return J1