import math
import numpy
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.linalg import toeplitz
from numpy import pi,arange,exp,sin,cos,zeros,tan,inf, dot, ones, sqrt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi,linspace,sin,exp,zeros,arange,real, ones, sqrt, array
from matplotlib.pyplot import figure, plot, show
from scipy.sparse import coo_matrix
from time import time
from Differentiation_Matrix import fourdif
from numba import jit
from scipy import optimize
from scipy.integrate import trapz
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

def f0(x):

    return np.exp(- 0.005 * x**2)

def f0_prime(x):

    return -0.01 * x * f0(x)

def F(z, x, t):

    return z + f0(z) * t - x

def fourier(k, z, y_hat, h, N):
    F = np.zeros(len(z))
    for j in range(0, len(z)):

        F[j] = sum(y_hat * np.exp(1j * k * h * j)).real / N
    return F

def exact_sol(x, t):
    exact = np.zeros([len(t), len(x)])
    exact[0, :] = f0(x)
    for i in range(1, len(t)):
        Z = np.zeros(len(x))
        for j in range(len(x)):
            zj = optimize.root(F, 0.0, args=(x[j], t[i]), tol=10 ** -10)
            Z[j] = zj.x

        u = f0(Z)
        exact[i, :] = u
    return exact


def euler(v, ik, p, dt):
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

    return w


def Simulation(N, xL, xR, tmax, dt):
    # Grid
    h = 2.0 * np.pi / N  # step size
    p = 2 * pi / (xR - xL)
    x = h * arange(0, N)
    x = x / p + xL

    # Initial conditions
    t = 0
    v = f0(x)

    # waves coefficients
    k = np.zeros(N)
    k[0: int(N / 2)] = np.arange(0, int(N / 2))
    k[int(N / 2) + 1:] = np.arange(-int(N / 2) + 1, 0, 1)
    ik = 1j * k
    k2 = ik ** 2

    # Setting up Plot
    tplot = 0.5
    plotgap = int(round(tplot / dt))
    nplots = int(round(tmax / tplot))

    LX = len(x)
    exact = np.zeros([nplots + 1, LX])
    exact[0, :] = v

    aprox = np.zeros([nplots + 1, LX])
    aprox[0, :] = v
    tdata = np.zeros(nplots + 1)
    for i in range(1, nplots+1):

        for n in range(plotgap):
            t = t + dt
            v = euler(v, ik, p, dt)

        aprox[i, :] = v
        if np.isnan(v).any():
            break

        tdata[i] = t

    return x, tdata, aprox


xL = -60
xR = 60
tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))
#tmax = 16
dt = 0.001

Ns = 2**np.arange(4, 9)
Nmax = 512

hz = 2.0 * np.pi / Nmax
z = hz * np.arange(0, Nmax)
q = 2 * np.pi / (xR - xL)
z = z / q + xL
exact = exact_sol(z, np.arange(0, tmax + 0.5, 0.5))

fig, ax = plt.subplots(1, 2, figsize=(15, 10))

colors= ['m--', 'c--', 'b--', 'g--', 'r--']
legend = [r'$u(x)$']
ax[0].plot(z, exact[-1, :], 'b-')
for j in range(len(Ns)):
    legend = legend + ['N' + ' = ' + str(Ns[j])]
    X, tdata, aprox = Simulation(Ns[j], xL, xR, tmax, dt)

    k = np.zeros(Ns[j])
    k[0: int(Ns[j] / 2)] = np.arange(0, int(Ns[j] / 2))
    k[int(Ns[j] / 2) + 1:] = np.arange(-int(Ns[j] / 2) + 1, 0, 1)

    T = len(tdata) - 1
    ut_hat = np.fft.fft(aprox[T, :])

    ut = fourier(k, z, ut_hat, hz, Ns[j])

    Error = abs(ut - exact[T, :])
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
    name = 'Numerical_Solution_Inviscid_T' + '.png'
    plt.tight_layout()
    #plt.savefig(name)


N = 512
X, tdata, aprox = Simulation(N, xL, xR, tmax, dt)
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

name = 'Numerical_Solution_Inviscid' + '.png'
plt.tight_layout()
#plt.savefig(name)

plt.show()