from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi, linspace, sin, exp, zeros, arange, real, ones, sqrt, array
from matplotlib.pyplot import figure, plot, show
from scipy.sparse import coo_matrix
from time import time

import math
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.linalg import toeplitz
from numpy import pi, arange, exp, sin, cos, zeros, tan, inf, dot, ones, sqrt
from numba import jit

from scipy.signal import fftconvolve, choose_conv_method
from Exact_Burg import exact as exact_sol

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#from Exact_inviscid import exact_sol
#from Heat_equation import exact_sol as heat

@jit(nopython=True, parallel=True)
def u0(x):

    return np.exp(-0.05 * x**2)


def phi0(z):

    integ0, err0 = quad(u0, 0, z)

    return integ0

def euler(t, v0, alpha, ik, k2, p, dt):
    p = p / 2
    data = np.zeros([len(t), int(N / 2)])
    data[0, :] = u0(x[::2])
    v0_hat = np.fft.fft(v0)
    factor = 1.0 + dt * alpha * k2 * p ** 2
    for j in range(1, len(t)):
        v_hat = v0_hat * factor**j
        v_hat_deriv = p * ik * v_hat
        data[j, :] = -(2.0 * alpha) * np.real(np.fft.ifft(v_hat_deriv) / np.fft.ifft(v_hat))[0:int(N / 2)]

    return data

def backward_euler(t, v0, alpha, ik, k2, p, dt):
    p = p / 2
    data = np.zeros([len(t), int(N / 2)])
    data[0, :] = u0(x[::2])
    v0_hat = np.fft.fft(v0)
    factor = 1.0 - dt * alpha * k2 * p ** 2
    for j in range(1, len(t)):
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        with np.errstate(divide='ignore', invalid='ignore'):
            v_hat = np.divide(v0_hat, np.power(factor, j))
            #data[j, :] = np.real(np.fft.ifft(v_hat))
            v_hat_deriv = p * ik * v_hat
            upper = np.fft.ifft(v_hat_deriv)
            lower = np.fft.ifft(v_hat)
            data[j, :] = - (2.0 * alpha) * np.real(np.divide(upper, lower))[0:int(N / 2)]

    return data

'''''
xL = -60
xR = 60
#tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))
tmax = 20.0

dt = 0.001
alpha = 20.0


xL = -60
xR = 60
#tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))
tmax = 80.0
dt = 0.0001
alpha = 1.0

Ns = 2**np.arange(6, 11)
Nmax = 1024
M = 256

hz = 2.0 * np.pi / M
z = hz * np.arange(0, M)
q = 2 * np.pi / (xR - xL)
z = z / q + xL
exact = exact_sol(alpha, z, tmax)
#x, tdata, exact = Simulation(Nmax, xL, xR, tmax, 0.001, alpha)

fig, ax = plt.subplots(1, 2, figsize=(15, 10))

colors= ['m--', 'c--', 'b--', 'g--', 'r--']
legend = [r'$u(x)$']

ax[0].plot(z, exact[-1, :], 'b-')
for j in range(len(Ns)):
    legend = legend + ['N' + ' = ' + str(Ns[j])]
    X, tdata, aprox = Simulation(Ns[j], xL, xR, tmax, dt, alpha)

    k = np.zeros(Ns[j])
    k[0: int(Ns[j] / 2)] = np.arange(0, int(Ns[j] / 2))
    k[int(Ns[j] / 2) + 1:] = np.arange(-int(Ns[j] / 2) + 1, 0, 1)

    ut_hat = np.fft.fft(aprox[-1, :])

    ut = fourier(k, z, ut_hat, hz, Ns[j])

    Error = abs(ut - exact[-1, :])
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
    ax[0].grid()
    ax[1].grid()

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
    #name = 'Numerical_Solution_alpha=' + str(alpha) + '_T=100.png'
    plt.tight_layout()


N = 32
X, tdata, aprox = Simulation(N, xL, xR, tmax, dt, alpha)
#W = heat(alpha, X, tdata)

#plt.figure()
#plt.plot(X, aprox[-1, :])
#plt.plot(X, W[-1, :])


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

name = 'Exact_Solution' + '.png'
plt.tight_layout()
#plt.savefig(name)

plt.show()
'''''
