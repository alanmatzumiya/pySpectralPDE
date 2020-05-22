from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi,linspace,sin,exp,zeros,arange,real, ones, sqrt, array
from matplotlib.pyplot import figure, plot, show
from scipy.sparse import coo_matrix
from time import time

import math
import numpy as np
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.linalg import toeplitz
from numpy import pi,arange,exp,sin,cos,zeros,tan,inf, dot, ones, sqrt
from numba import jit
from scipy import signal
from scipy import optimize


def f0(x):

    return np.exp(-0.005 * x**2)

def f0_prime(x):

    return -0.01 * x * f0(x)


def euler(v, ik, k2, p, alpha, dt):
    error = 1.0
    tol = 10**-10
    v_hat = np.fft.fft(v)
    v_hat_old = v_hat

    while error > tol:
        Fv_old = W(v_hat_old, ik, p)

        v_hat_new = v_hat - dt * (Fv_old)

        error = max(abs(v_hat_new - v_hat_old))
        v_hat_old = v_hat_new

    return real(np.fft.ifft(v_hat_old))


def W(v_hat, ik, p):
    v2 = np.fft.ifft(v_hat)**2
    w = 0.5 * p * ik * np.fft.fft(v2)

    #w_hat = 0.5 * ik * p * v_hat**2
    #w_hat = ik * p * np.fft.fft(signal.fftconvolve(v_hat, v_hat, 'same'))
    #w_hat = np.fft.fft(signal.convolve(v, v[::-1], 'same'))
    return w  #0.5 * ik * p * w_hat


def Simulation(N, xL, xR, tmax, dt, alpha):
    # Grid
    h = 2.0 * np.pi / N  # step size
    p = 2 * pi / (xR - xL)
    x = h * arange(0, N)
    x = x/p + xL

    # Initial conditions
    t = 0
    v = f0(x)

    # waves coefficients
    k = np.zeros(N)
    k[0: int(N / 2)] = np.arange(0, int(N / 2))
    k[int(N / 2) + 1:] = np.arange(-int(N / 2) + 1, 0, 1)
    ik = 1j * k
    k2 = ik**2


    # Setting up Plot
    tplot = 0.5
    plotgap = int(round(tplot/dt))
    nplots = int(round(tmax/tplot))

    LX = len(x)
    data = np.zeros([nplots + 1, LX])
    data[0, :] = v
    tdata = np.zeros(nplots + 1)
    for i in range(1, nplots+1):

        for n in range(plotgap):  # RK4
            t = t + dt


            v = euler(v, ik, k2, p, alpha, dt)
            # RK4
            '''''
            a = p ** 2 * alpha * k2 * v_hat - W(v_hat, ik, p)
            b = p ** 2 * alpha * k2 * (v_hat + 0.5 * a * dt) - W(v_hat + 0.5 * a * dt, ik, p)
            c = p ** 2 * alpha * k2 * (v_hat + 0.5 * b * dt) - W(v_hat + 0.5 * b * dt, ik, p)
            d = p ** 2 * alpha * k2 * (v_hat + c * dt) - W(v_hat + c * dt, ik, p)

            v_hat = v_hat + dt * (a + 2 * (b + c) + d) / 6.0
            '''''
        data[i, :] = v
        if np.isnan(v).any():
            break
        # real time vector
        tdata[i] = t

    return tdata, data, x
N = 128
xL = -60
xR = 60

#pc = optimize.brenth(f0_biprime, -1, xR)
#tc = -1.0 / f0_prime(pc)
#print(tc)
tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))
tmax = 17
alpha = 0.1
dt = 0.01

ti = time()
tdata, data, X = Simulation(N, xL, xR, tmax, dt, alpha)


if np.isnan(data).any() == False:
    tf = time() - ti
    print(tf)
    plt.figure(1)
    plt.plot(X, data[len(tdata) - 1, :])
    print(max(data[len(tdata) - 1, :]))
    X, Y = np.meshgrid(X, tdata)
    Z = data
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.gca(projection='3d')
    print(len(tdata))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.1)

    ax.set_zlim(0, 1.0)
    ax.set_xlabel(r'$x$', fontsize=15)
    ax.set_ylabel(r'$t$', fontsize=15)
    ax.set_zlabel(r'$u$', fontsize=15)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
