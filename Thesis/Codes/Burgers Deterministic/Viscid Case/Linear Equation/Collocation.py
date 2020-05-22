import matplotlib.pyplot as plt

from matplotlib import cm

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi, round

from time import time
from Differentiation_Matrix import fourdif
from matplotlib.ticker import LinearLocator, FormatStrFormatter


@jit(nopython=True, parallel=True)
def u0(x):

    return np.exp(-0.05 * x**2)


def phi0(z):

    integ0, err0 = quad(u0, 0, z)

    return integ0

def euler(t, v0, alpha, D1, D2, p, dt):
    p = p / 2
    data = np.zeros([len(t), int(N / 2)])
    data[0, :] = u0(x[::2])
    factor = 1.0 + dt * alpha * D2 * p ** 2
    for j in range(1, len(t)):
        v = v0 * factor**j
        v_deriv = p * D1 * v
        data[j, :] = - 2.0 * alpha * (v_deriv / v)[0:int(N / 2)]

    return data

def backward_euler(t, v0, alpha, D1, D2, p, dt):
    p = p / 2
    data = np.zeros([len(t), int(N / 2)])
    data[0, :] = u0(x[::2])
    factor = 1.0 - dt * alpha * D2 * p ** 2
    for j in range(1, len(t)):
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.divide(v0, np.power(factor, j))
            #data[j, :] = np.real(np.fft.ifft(v_hat))
            v_deriv = p * ik * v
            data[j, :] = - 2.0 * alpha * (np.divide(v_deriv, v)[0:int(N / 2)]

    return data
