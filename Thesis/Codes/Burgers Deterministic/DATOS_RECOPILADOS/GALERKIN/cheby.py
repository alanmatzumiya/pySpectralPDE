import math
import numpy as np


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
        c = 2.0/N * np.dot(y,T)
        c[0] = c[0]/2
        return Cheby(a,b,*c)
