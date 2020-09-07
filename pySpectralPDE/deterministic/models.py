from numpy import zeros, fft, dot, argmax, real
from numba import jit
from .tools import differentation

class equations:

    def __init__(self):

    @staticmethod
    @jit(target='cpu', nopython=True)
    def diffusion(nu, diff2):

        return nu * diff_2

    @staticmethod
    @jit(target='cpu', nopython=True)
    def lineal_convection(c, diff_1):

        return c * diff_1 

    @staticmethod
    @jit(target='cpu', nopython=True)
    def nonlinear_convection(diff_1, v=None):
        if v is not None:
            return v * diff_1
        else:
            return 0.5 * diff_1 

    @staticmethod
    @jit(target='cpu', nopython=True)
    def burgers(nu, diff_1, diff_2, v=None):

        return nonlinear_convection(diff_1, v) + diffusion(nu, diff_2)


