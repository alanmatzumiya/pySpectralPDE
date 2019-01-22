from numpy import sin, pi, sqrt, exp, cos
import numpy as np
from math import factorial
import scipy.integrate as integrate
from scipy import special


def u0(x):
    """
    Initial Condition

    Parameters
    ----------
    x : array or float;
        Real space

    Returns
    -------
    array or float : Initial condition evaluated in the real space

    """
    return (1.0 /np.cosh(5.0 * (x - 0.5)))**2


def evalXSin(nu, k, u0):
    """
    Function to calculate inner product between initial condition
    and the complete orthonormal system of eigenfunctions e_k

    Parameters
    ----------
    nu : float; Diffusion coefficient
    k : int; index of eigenfunction
    u0 : callable(x); Initial Condition

    Returns
    -------
    int: float, integrals value

    """
    f = lambda x: sqrt(2.0 * nu) * k * pi * sqrt(2.0 / pi) * (u0(x)) * (sin(k * pi * x))

    integr = integrate.quad(f, 0, 1)  ### 1

    int = integr[0]

    return int

