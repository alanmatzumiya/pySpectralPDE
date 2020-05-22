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

from scipy.signal import fftconvolve
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def f0(x):

    return np.exp(-0.005 * x**2)

def G(x, t, alpha):

    eta = 4.0 * alpha * t

    return np.exp(- x**2 / eta)

def fourier(k, z, y_hat, h, N):
    F = np.zeros(len(z))
    for j in range(0, len(z)):

        F[j] = sum(y_hat * np.exp(1j * k * h * j)).real / N
    return F

# using the analytic solution:
def exact_sol(alpha, x, t):

    exact = np.zeros([len(t), len(x)])
    exact[0, :] = f0(x)

    Q = 100
    rule1 = np.polynomial.hermite_e.hermegauss(Q)
    rulesX = rule1[0][::-1]
    rulesW = rule1[1]

    for j in range(0, len(x)):
        xj = x[j]

        for i in range(1, len(t)):
            ti = t[i]

            eps = (1.0 / np.sqrt(4.0 * alpha * ti * np.pi))
            factor = np.sqrt(2.0 * alpha * ti)
            z = xj - rulesX * factor
            sum1 = 0.0
            for k in range(len(z)):
                sum1 = sum1 + factor * f0(z[k]) * rulesW[k]
            exact[i, j] = eps * sum1
            '''''
            h = 0.1
            y1 = np.arange(-M, M + h, h)
            ti = t[i]
                
            eps = (1.0 / np.sqrt(4.0 * alpha * ti * np.pi))
        
            sum1 = 0.0
            for k in range(1, len(y1) - 1):
                sum1 = sum1 + G(xj - y1[k], ti, alpha) * f0(y1[k])

            Ga = G(xj - y1[0], ti, alpha) * f0(y1[0])
            Gb = G(xj - y1[len(y1) - 1], ti, alpha) * f0(y1[len(y1) - 1])
            sum1 = h * (sum1 + 0.5 * (Ga + Gb))
      
            exact[i, j] = eps * sum1
            '''''
    return exact


def euler(v, k2, p, alpha, dt):
    v_hat = np.fft.fft(v)
    Fv_hat = alpha * k2 * p**2
    #v_hat = v_hat / (1.0 - dt * Fv_hat)
    v_hat = v_hat + dt * Fv_hat * v_hat
    v = real(np.fft.ifft(v_hat))

    return v

def A(v0_hat, k2, p, alpha, t):

    Gk = np.exp(p**2 * alpha * k2 * t)
    v_hat = v0_hat * Gk
    v = real(np.fft.ifft(v_hat))

    return v


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
    aprox = np.zeros([nplots + 1, LX])
    aprox[0, :] = v
    tdata = np.zeros(nplots + 1)
    for i in range(1, nplots+1):

        for n in range(plotgap):  # RK4
            t = t + dt

            v = euler(v, k2, p, alpha, dt)

        aprox[i, :] = v
        if np.isnan(v).any():
            break
        # real time vector
        tdata[i] = t

    return x, tdata, aprox



xL = -60
xR = 60
#tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))
tmax = 100.0
dt = 0.001
alpha = 1.0

Ns = 2**np.arange(4, 7)
Nmax = 1024

hz = 2.0 * np.pi / Nmax
z = hz * np.arange(0, Nmax)
q = 2 * np.pi / (xR - xL)
z = z / q + xL

k = np.zeros(Nmax)
k[0: int(Nmax / 2)] = np.arange(0, int(Nmax / 2))
k[int(Nmax / 2) + 1:] = np.arange(-int(Nmax / 2) + 1, 0, 1)
ik = 1j * k
k2 = ik**2

#exact = exact_sol(alpha, z, np.arange(0, tmax + 0.5, 0.5))

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
exact = A(np.fft.fft(f0(z)), alpha, q, k2, tmax)

colors= ['m--', 'r--', 'g--', 'b--', 'c--']
legend = [r'$u(x)$']
ax[0].plot(z, exact, 'b-')
for j in range(len(Ns)):
    legend = legend + ['N' + ' = ' + str(Ns[j])]
    X, tdata, aprox = Simulation(Ns[j], xL, xR, tmax, dt, alpha)

    k = np.zeros(Ns[j])
    k[0: int(Ns[j] / 2)] = np.arange(0, int(Ns[j] / 2))
    k[int(Ns[j] / 2) + 1:] = np.arange(-int(Ns[j] / 2) + 1, 0, 1)

    T = len(tdata) - 1
    ut_hat = np.fft.fft(aprox[T, :])

    ut = fourier(k, z, ut_hat, hz, Ns[j])

    Error = abs(ut - exact)
    # ax[0].plot(s_1, func(s_1, *const1), '--')
    ax[0].plot(z, ut, colors[j], linewidth=1.5)
    ax[1].semilogy(z, Error, colors[j], linewidth=1.5)

    ax[0].set_xlim(xL, xR)
    ax[1].set_xlim(xL, xR)
    ax[0].set_xticks(np.linspace(xL, xR, 7))
    ax[1].set_xticks(np.linspace(xL, xR, 7))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax[0].set_xlabel(r'$x$', fontsize=25, color='black')
    ax[0].set_ylabel(r'\textbf{$u_N$}',
                     fontsize=25, color='black')
    ax[1].set_xlabel(r'$x$', fontsize=25, color='black')
    ax[1].set_ylabel(r'\textbf{$| u - u_N |$}',
                     fontsize=25, color='black')

    params = {'legend.fontsize': 25,
              'legend.handlelength': 0.5}
    plt.rcParams.update(params)

    ax[0].text(-0.1, 1.08, '(a)',
               horizontalalignment='left', fontsize=25,
               transform=ax[0].transAxes)
    ax[1].text(1.08, 1.08, '(b)',
               horizontalalignment='left', fontsize=25,
               transform=ax[0].transAxes)

    ax[0].legend(legend)
    ax[0].tick_params(axis='x', labelsize='25')
    ax[1].tick_params(axis='x', labelsize='25')
    ax[0].tick_params(axis='y', labelsize='25')
    ax[1].tick_params(axis='y', labelsize='25')
    name = 'Numerical_Solution_Heat_T' + '.png'
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig(name)

N = 128
X, tdata, aprox = Simulation(N, xL, xR, tmax, dt, alpha)
X, Y = np.meshgrid(X, tdata)
Z = aprox

fig1 = plt.figure(figsize=(15, 15))
ax1 = fig1.gca(projection='3d')

surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.Spectral_r, rstride=1, cstride=1)

# ax2.contour3D(X, Y, W, cmap='binary')

ax1.set_zlim(0, 1.0)
ax1.set_xlabel(r'$x$', fontsize=25, labelpad=20)
ax1.set_ylabel(r'$t$', fontsize=25, labelpad=20)
ax1.set_zlabel(r'$u(x, t)$', fontsize=25, labelpad=30)
ax1.view_init(30, -80)

ax1.tick_params(axis='x', labelsize='25', pad=5)
ax1.tick_params(axis='y', labelsize='25', pad=5)
ax1.tick_params(axis='z', labelsize='25', pad=12)

name = 'Numerical_Solution_Heat' + '.png'
plt.tight_layout()
plt.savefig(name)

plt.show()
