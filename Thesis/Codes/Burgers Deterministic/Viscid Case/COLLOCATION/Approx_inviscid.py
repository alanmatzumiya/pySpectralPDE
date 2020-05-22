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

def chebfft(v):
    '''Chebyshev differentiation via fft.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
    '''
    N = len(v)-1
    if N == 0:
        w = 0.0 # only when N is even!
        return w
    x  = cos(pi*arange(0,N+1)/N)
    ii = arange(0,N)
    V = np.flipud(v[1:N]); V = list(v) + list(V);
    U = real(np.fft.fft(V))
    b = list(ii); b.append(0); b = b + list(arange(1-N,0));
    w_hat = 1j*array(b)
    w_hat = w_hat * U
    W = real(np.fft.ifft(w_hat))
    w = zeros(N+1)
    w[1:N] = -W[1:N]/sqrt(1-x[1:N]**2)
    w[0] = sum(ii**2*U[ii])/N + 0.5*N*U[N]
    w[N] = sum((-1)**(ii+1)*ii**2*U[ii])/N + \
              0.5*(-1)**(N+1)*N*U[N]
    return w




def euler(v, ik, k2, p, alpha, dt):
    error = 1.0
    tol = 10**-10
    v_max = 1.0
    #v_hat = np.fft.fft(v)
    v_old = v
    #R = (0.5 * v_max - v) * p * chebfft(v)
    #R[0] = 0; R[len(v) - 1] = 0
    #v_hat_old = v_hat
    #Fv_old = Wc(v_old, v_max, ik, p)
    while error > tol:

        #v_hat_new = v_hat + dt * (Fv_hat - 0.5 * v_max * p * chebfft(v_hat_old))
        Fv_old = Wc(v_old, v_max, ik, p)
        #v_new = v - dt * (Fv_old - R)
        v_new = v - dt * (Fv_old)

        #W_prime = p * ik * Fv_hat_old
        #v_hat_new = v_hat + dt * Fv_hat_old
        #W_prime = p * ik * Fv_hat_old

        #v_prime = p * ik * v_hat_old
        #Wv_hat = np.fft.fft(np.fft.ifft(v_prime) * np.fft.ifft(W_prime))
        #v_hat_new = v_hat_old -(v_hat_old - v_hat - dt * Fv_hat_old) / (1 - dt * Wv_hat)
        error = max(abs(v_new - v_old))
        v_old = v_new

    return v_old #real(np.fft.ifft(v_hat_new))


def W(v, v_max, ik, p):
    w = 0.5 * v_max * p * chebfft(v)
    w[0] = 0; w[len(v) - 1] = 0
    #w_hat = 0.5 * ik * p * v_hat**2
    #w_hat = ik * p * np.fft.fft(signal.fftconvolve(v_hat, v_hat, 'same'))
    #w_hat = np.fft.fft(signal.convolve(v, v[::-1], 'same'))
    return w  #0.5 * ik * p * w_hat

def Wc(v, v_max, ik, p):

    w = 0.5 * p * chebfft(v**2)
    w[0] = 0; w[len(v) - 1] = 0

    #w_hat =chebfft(w)
    #w_hat = 0.5 * ik * p * v_hat**2
    #w_hat = ik * p * np.fft.fft(signal.fftconvolve(v_hat, v_hat, 'same'))
    #w_hat = np.fft.fft(signal.convolve(v, v[::-1], 'same'))
    return w  #0.5 * ik * p * w_hat


def Simulation(N, xL, xR, tmax, dt, alpha):
    # Grid

    x = cos(arange(0, N + 1) * pi / N)
    p = 2.0 / (xR - xL)
    x = (1 / p) * (x - 1.0) + xR

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
N = 1024
xL = -60
xR = 60
tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))

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
