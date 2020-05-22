import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.integrate import quad, dblquad, romberg, trapz
from scipy.special import erf, eval_hermitenorm
from numpy import polynomial
import math
from time import time
from numba import jit, njit

from matplotlib.ticker import LinearLocator, FormatStrFormatter


def analytic(t, a, b, v0, alpha):

    phi_0 = np.zeros(len(x))
    for j in range(0, len(x)):
        phi_0[j] = np.exp(- phi0(x[j]) / (2.0 * alpha))

    phi_hat = np.fft.fft(phi_0[::2])
    solutions = np.zeros([len(t), len(x)])
    solutions[0, :] = u0(x)

    for j in range(1, len(t)):

        Gk = np.exp(p**2 * alpha * k2 * t[j])
        phit_hat = phi_hat * Gk * N
        phit_deriv = np.real(np.fft.ifft(p * ik * phit_hat))
        phit = np.real(np.fft.ifft(phit_hat))
        solutions[j, :] = - 2.0 * alpha * phit_deriv[::-1] / phit[::-1]


def numerical(t, x):
    # waves coefficients
    k = np.zeros(N)
    k[0: int(N / 2)] = np.arange(0, int(N / 2))
    k[int(N / 2):] = np.arange(-int(N / 2), 0)
    ik = 1j * k
    k2 = ik ** 2

    phi_0 = np.zeros(len(x))
    I = []
    for xj in x:
        I.append(np.exp(- phi0(xj) / (2.0 * alpha)))

    I = I[::2]
    phi_0[0: int(N / 2)] = I;
    phi_0[int(N / 2):] = I[::-1]

    data = backward_euler(t, phi_0, alpha, k2, p, dt)

    return data

def grapher(x, t, data):

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(x[::2], t)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.1)

    ax.set_zlim(-1, 1)
    ax.set_xlabel(r'$x$', fontsize=15)
    ax.set_ylabel(r'$t$', fontsize=15)
    ax.set_zlabel(r'$u$', fontsize=15)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


# parameters
a = -60; b = 60
N = 64
hx = 2.0 * np.pi / N # step size
p = 2.0 * np.pi / (b - a)
x = hx * np.arange(0, N)
x = x / p + a
dt = 0.01 # grid size for time (s)
alpha = 0.001 # kinematic viscosity of oil (m2/s)
t_max = 100.0 # total time in s
t = np.arange(0, t_max + dt, dt)

