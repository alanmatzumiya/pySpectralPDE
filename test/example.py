from pySpectralPDE import spectralPDE
import numpy as np


def u0(z):
    return np.exp(- 5.0 ** (-3) * z ** 2)


params = dict(nu=1.0, N=32, xL=-60.0, xR=60.0, dt=0.01, t0=0.0, tmax=100.0)
solver = spectralPDE.setup_solver(u0, params)

