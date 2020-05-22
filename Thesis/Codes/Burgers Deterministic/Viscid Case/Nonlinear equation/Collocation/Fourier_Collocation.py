import matplotlib.pyplot as plt

from matplotlib import cm

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi, round

from time import time
from Differentiation_Matrix import fourdif
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def f0(x):

    return np.exp(- 0.05 * x ** 2)


# time numerical integrators
def forward_euler(alpha, v, dt, p, D1, D2):
    v = v + dt * (p ** 2 * alpha * np.dot(D2, v) - 0.5 * p * np.dot(D1, v**2))

    return v


def backward_euler(alpha, v_old, dt, p, D1, D2):
    v_test = forward_euler(alpha, v_old, dt, p, D1, D2)
    L = v_old + p ** 2 * alpha * np.dot(D2, v_old) * dt

    error = 1.0
    tol = 1.0 ** -10
    while error > tol:
        v_new = L - dt * 0.5 * p * np.dot(D1, v_test**2)
        error = max(abs(v_new - v_test))

        v_test = v_new

    return v_test


def Simulation(N, xL, xR, tmax, dt, alpha):
    # Grid
    x, D1 = fourdif(N, 1)
    x, D2 = fourdif(N, 2)

    # scaling
    p = 2.0 * pi / (xR - xL)
    x = x / p + xL

    # Initial conditions
    t = 0
    v = f0(x)

    # Setting up Plot
    tplot = 0.5
    plotgap = int(round(tplot / dt))
    nplots = int(round(tmax / tplot))

    LX = len(x)
    data = np.zeros([nplots + 1, LX])
    data[0, :] = v
    tdata = np.zeros(nplots + 1)

    for i in range(1, nplots + 1):

        for n in range(plotgap):
            t = t + dt
            #v = forward_euler(alpha, v, dt, p, D1, D2)
            v = backward_euler(alpha, v, dt, p, D1, D2)

        data[i, :] = v
        if np.isnan(v).any():
            break
        # real time vector
        tdata[i] = t

    return tdata, x, data