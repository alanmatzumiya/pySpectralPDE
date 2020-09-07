from numba import jit
from numpy import pi, sqrt, exp
import numpy as np
from math import factorial
from scipy.integrate import dblquad
from scipy import special
import itertools
import math



def chebspace(npts):
    """
    function to create chebycheb nodes

    Parameters
    ----------
    npts : int; points number

    Returns
    -------
    points: array, chebycheb nodes

    """
    t = (np.array(range(0, npts)) + 0.5) / npts
    return -np.cos(t*math.pi)


def chebmat(u, N):
    T = np.column_stack((np.ones(len(u)), u))
    for n in range(2,N+1):
        Tnext = 2*u*T[:,n-1] - T[:,n-2]
        T = np.column_stack((T,Tnext))
    return T


class Cheby(object):
    """
     This module approximate the function with chebyshev polynomials

    """

    def __init__(self, a, b, *coeffs):
        self.c = (a+b)/2.0
        self.m = (b-a)/2.0
        self.coeffs = np.array(coeffs, ndmin=1)
    def rangestart(self):
        return self.c-self.m
    def rangeend(self):
        return self.c+self.m
    def range(self):
        return (self.rangestart(), self.rangeend())
    def degree(self):
        return len(self.coeffs)-1
    def truncate(self, n):
        return Cheby(self.rangestart(), self.rangeend(), *self.coeffs[0:n+1])
    def asTaylor(self, x0=0, m0=1.0):
        n = self.degree()+1
        Tprev = np.zeros(n)
        T     = np.zeros(n)
        Tprev[0] = 1
        T[1]  = 1
        # evaluate y = Chebyshev functions as polynomials in u
        y     = self.coeffs[0] * Tprev
        for co in self.coeffs[1:]:
            y = y + T*co
            xT = np.roll(T, 1)
            xT[0] = 0
            Tnext = 2*xT - Tprev
            Tprev = T
            T = Tnext
        # now evaluate y2 = polynomials in x
        P     = np.zeros(n)
        y2    = np.zeros(n)
        P[0]  = 1
        k0 = -self.c/self.m
        k1 = 1.0/self.m
        k0 = k0 + k1*x0
        k1 = k1/m0
        for yi in y:
            y2 = y2 + P*yi
            Pnext = np.roll(P, 1)*k1
            Pnext[0] = 0
            P = Pnext + k0*P
        return y2
    def __call__(self, x):
        xa = np.array(x, copy=False, ndmin=1)
        u = np.array((xa-self.c)/self.m)
        Tprev = np.ones(len(u))
        y = self.coeffs[0] * Tprev
        if self.degree() > 0:
            y = y + u*self.coeffs[1]
            T = u
        for n in range(2,self.degree()+1):
            Tnext = 2*u*T - Tprev
            Tprev = T
            T = Tnext
            y = y + T*self.coeffs[n]
        return y
    def __repr__(self):
        return "Cheby%s" % (self.range()+tuple(c for c in self.coeffs)).__repr__()
    @staticmethod
    def fit(func, a, b, degree):
        """
        function to approximate func

        Parameters
        ----------
        func : callable(x); function to approximate
        a : float; lower lim
        b : float; upper lim
        degree : int; grade of the polynomial

        Returns
        -------
        Cheby: callable(x), approximated function

        """
        N = degree+1
        u = chebspace(N)
        x = (u*(b-a) + (b+a))/2.0
        y = func(x)
        T = chebmat(u, N=degree)
        c = 2.0/N * np.dot(y, T)
        c[0] = c[0]/2
        return Cheby(a, b, *c)


class JNM:

    @staticmethod
    def J_NM(N, M, N1):
        """
        function to construct the J^{N;M} polynomial products by column.

        Parameters
        ----------
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
            J[0, L1 - 1:L2] = 1
            J[1, L1 - 1:L2] = 2
            for j in range(0, L):
                for i in range(0, len(h[j])):
                    J[2 + i, L1 + j - 1] = h[j][i]
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

    def Js(self, N):
        """
        Function to compute set J

        Returns
        -------
        J1 : array, shape(len(M), len(M))
            Array containing the grade of each Hermite polynomial

        """
        if (N == 4):
            M = 7
            N1 = 3
            return self.J_NM(N, M, N1)

        if (N == 5):
            M = 11
            N1 = 4
            return self.J_NM(N, M, N1)

        if (N == 6):
            M = 18
            N1 = 4
            J2 = self.J_NM(N, M, N1)
            SeqJ = np.array([1, 3] + list(range(5, 7)) + [8] + list(range(11, 14))
                            + [15, 17])  #### This is with the numbers (N=6, M=18, N1=4),
            for i in range(len(SeqJ)):
                SeqJ[i] = SeqJ[i] - 1
            M = 10
            return J2[0:M, SeqJ]

        if (N == 7):
            M = 29
            N1 = 4
            J2 = self.J_NM(N, M, N1)
            SeqJ = np.array(list(range(1, 10)) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28,
                                                  29])  #### This is with the numbers (N=7, M=29, N1=4),
            for i in range(len(SeqJ)):
                SeqJ[i] = SeqJ[i] - 1
            M = 20
            return J2[0:M, SeqJ]

        if (N == 8):
            M = 60
            N1 = 5
            J2 = self.J_NM(N, M, N1)
            SeqJ = np.array(
                list(range(1, 10)) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29, 31, 34, 35, 38, 39, 44, 48, 51, 55,
                                      59])  #### This is with the numbers (N=7, M=29, N1=4),
            for i in range(len(SeqJ)):
                SeqJ[i] = SeqJ[i] - 1
            M = 30
            return J2[0:M, SeqJ]

        if (N == 9):
            M = 150
            N1 = 5
            J2 = self.J_NM(N, M, N1)
            SeqJ = np.array(
                list(range(1, 10)) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29, 31, 34, 35, 38, 39, 44, 48, 51, 55,
                                      59, 63, 67, 70, 75, 81, 85, 89, 92, 95,
                                      99])  #### This is with the numbers (N=7, M=29, N1=4),
            for i in range(len(SeqJ)):
                SeqJ[i] = SeqJ[i] - 1
            M = 40
            return J2[0:M, SeqJ]





class analysis:

    def __init__(self, N):

        self.J = JNM().Js(N)
        self.M = len(self.J[:, 1])

    @staticmethod
    def integ(f, Jlj):
        """
        This function calculates the integral of the norm between two solutions
        with different initial conditions for a fixed point in real space

        Returns
        -------
        Pn, Qn : array, shape(M, M)
            array containing the integrals of the Hermite polynomials

        """
        @jit
        def measure(z):
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
            return (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z ** 2)

        return dblquad(f, measure(-np.inf), measure(np.inf),
                       lambda x: measure(-np.inf), lambda x: measure(np.inf), args=[Jlj])[0]

    @jit
    def distance(self, ux, uy):
        """
        This function calculates the norm between two solutions
        with different initial conditions for a fixed point in real space

        Parameters
        ----------
        Utn : array, shape(M, len(xSpace))
            Array containing the ordinary differential equations system constants to u0
        Vtn : array, shape(M, len(xSpace))
            Array containing the ordinary differential equations system constants to approximation

        Returns
        -------
        norms : array; norms between two solutions
        times : array; discretized time
        """

        def f1(x, y, Jij):
            return abs(
                (-1) ** Jij * (special.eval_hermitenorm(Jij, x)
                               - special.eval_hermitenorm(Jij, y)) / factorial(Jij + 1) ** 2
            ) ** 2

        def f2(x, y, Jij):
            return abs(
                (-1) ** Jij * special.eval_hermitenorm(Jij, y) / factorial(Jij + 1) ** 2
            ) ** 2

        norms = np.zeros(len(ux[:, 0]))
        for k in range(0, len(norms)):
            for j in range(0, self.M):
                for i in range(0, self.M):
                    prod1, prod2 = 1.0, 1.0
                    for l in range(0, self.M):
                        if self.J[l, j] > 0:
                            prod1 = prod1 * self.integ(f1, int(self.J[l, j]))
                            prod2 = prod2 * self.integ(f2, int(self.J[l, j]))
                    norms[k] = norms[k] + (
                            ux[k, self.J[i, j]] - uy[k, self.J[i, j]]) ** 2 * prod1 + ux[self.J[i, j]] ** 2 * prod2
        return norms