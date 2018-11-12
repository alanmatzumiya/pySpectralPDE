from numpy import sin, pi, sqrt, exp, cos
import numpy as np
from scipy import special
from Integration import integr2, integr4


def u02(J, M, rulesX, rulesW, LRules, xSpace, nu, u0):
    """
    Function to calculate the constants of the system of ordinary differential equations given the initial condition

    Parameters
    ----------
    J : array; shape(M, M)
        Array containing the grades of the polynomials
    N : int; order max of the polynomials
    M : int; number of ODEs
    rulesX : array; polynomials of Hermite evaluated
    rulesW : array; weights of polynomials of Hermite evaluated
    LRules : array; length of rulesX array
    xSpace : array; discretized real space
    nu : array; Diffusion coefficient
    u0 : callable(x); initial condition function

    Returns
    -------
    ci: array, shape(M, len(xSpace))
        Array containing the constants of the system of ordinary differential equations

    """

    Lx = len(xSpace)
    ci = np.zeros([M,Lx])

    for z in range(0, Lx):
        for i in range(0, M):
            sum2 = 0
            for k in range(0, M):
                sum1 = 0
                if(J[k,i]>0):
                    for y in range(0, LRules):
                        sum1 = sum1 + special.eval_hermitenorm(int(J[k,i]), rulesX[y]) * rulesX[y] * rulesW[y]
                sum2 = sum2 + sum1 * u0(xSpace[z]) * sqrt(2.0)/((sqrt(2.0 * nu) * pi * (k + 1)))
            ci[i,z] = sum2
    return ci


def Cnm(J, N, M, rulesX, rulesW, LRules, nu):
    """
    Function to computes Matrix

    Parameters
    ----------
    J : array; shape(M, M)
    Array containing the grades of the polynomials
    M : int; max order polynomials
    rulesX : array; polynomials of Hermite evaluated
    rulesW : array; weigths of polynomials of Hermite evaluated
    LRules : array; length of rulesX array
    nu : array; Diffusion coefficient

    Returns
    -------
    Cnm: array, shape(len(t), len(y0))
    Array containing the value of y for each desired time in t,

    """
    Cnm = np.zeros([M, M])
    I4 = integr4(M)
    for k in range(1, M):
        for i in range(1, M):
            I2 = integr2(N, J[:,k],J[:,i],M,rulesX,rulesW,LRules,nu, k, i)
            sum1 = 0
            for j in range(0, M):
                sum1 = sum1 + (j + 1) * sqrt(J[j, i]) * I4[j] * I2[j]
            Cnm[k, i] = sum1
    Cnm[0, 0] = pi * (-3.0) * I4[0]
    return Cnm


def ALambda1(Cs1, J, M, nu):
    """
    Function to calculate the matrix of the system of ordinary differential equations.

    Parameters
    ----------
    Cs1 : array; discretized pressure field
    J : int; number of timesteps to run
    M : int; size of individual timestep
    nu : array; diffusive coefficient

    Returns
    -------
    Alambda1: array, shape(M, M)
    Matrix of ordinary differential equations system
    """
    Lamb1 = np.zeros([M, M])
    for i in range(0, M):
        sum1 = 0
        for j in range(0, M):
            sum1 = sum1 + J[j, i] * ((j + 1)**2) * pi**2

        Lamb1[i, i] = sum1 * nu
    ALambda1 = Cs1 - Lamb1

    return ALambda1


def EigeF(J, N, M, rulesX, rulesW, LRules, xSpace, nu, u0):

    """
    Function to computes eigenvalues and eigenvectors of the matrix A

    Parameters
    ----------
    J : array; shape(M, M)
        Array containing the grades of the polynomials
    N : int; order max of the polynomials
    M : int; number of ODEs
    rulesX : array; polynomials of Hermite evaluated
    rulesW : array; weights of polynomials of Hermite evaluated
    LRules : array; length of rulesX array
    xSpace : array; discretized real space
    nu : array; Diffusion coefficient
    u0 : callable(x); initial condition function

    Returns
    -------
    EigValRe: array, size M
        Array containing the real eigenvalues of matrix A
    EigValIm: array, size M
        Array containing the imaginary eigenvalues of matrix A
    EigVecRe: array, shape(M, M)
        Array containing the real eigenvectors of matrix A
    EigVecIm: array, shape(M, M)
        Array containing the imaginary eigenvectors of matrix A
    U_1: array, shape(M, len(xSpace))
        Array containing the ordinary differential equations system constants

    """
    Cs = Cnm(J, N, M, rulesX, rulesW, LRules, nu)
    A = ALambda1(Cs, J, M, nu)
    Eig1 = np.linalg.eig(A)
    EigValRe = Eig1[0].real
    EigValIm = Eig1[0].imag
    EigVecRe = Eig1[1].real
    EigVecIm = Eig1[1].imag

    U_1 = u02(J, M, rulesX, rulesW, LRules, xSpace, nu, u0)

    return EigValRe, EigValIm, EigVecRe, EigVecIm, U_1