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
from scipy.integrate import trapz
from Exact_Burg import exact as exact_sol

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def f0(x):

    return np.exp(- 0.005 * x**2)

def fourier(z, ut, N, x, h):
    F = np.zeros(len(z))

    ns = np.arange(-int(N/2), int(N/2) + 1, 1)

    un = np.zeros(N, dtype='complex')

    for j in range(0, N):
            y = ut * np.exp(-1j * ns[j] * x) / (2.0 * np.pi)
            un[j] = trapz(y, x)
    for i in range(0, len(z)):
            if np.allclose(h * i, z[i]):
                F[i] = ut[i]
            else:
                F[i] = sum(un * np.exp(1j * ns * h * i)).real
    return F

@jit
def euler(v, D1, D2, dt, alpha, L):
    tol = 10**-10
    error = 1.0
    v_old = v
    while error > tol:

        #v_new = v + dt * (L ** 2 * alpha * np.dot(D2, v) - L * v_old * np.dot(D1, v_old))
        #F = v + dt * (L ** 2 * alpha * np.dot(D2, v) - 0.5 * L * np.dot(D1, v_old**2)) - v_old
        #F_prime = dt * (L**2 * np.dot(D1, np.dot(D1, v_old**2))) - 1.0
        #v_new = v_old - F / F_prime
        v_new = v + dt * (L ** 2 * alpha * np.dot(D2, v) - 0.5 * L * np.dot(D1, v_old ** 2))
        error = max(abs(v_new - v_old))
        v_old = v_new
    return v_old


def Simulation(N, xL, xR, tmax, dt, alpha):
    # Grid
    x, D1 = fourdif(N, 1)
    x, D2 = fourdif(N, 2)

    # scaling
    #xL = xL / np.sqrt(alpha)
    #xR = xR / np.sqrt(alpha)
    #print(xL)
    L = 2.0 * pi / (xR - xL)
    x = x/L + xL
    #x = np.sqrt(alpha) * x
    #x = x * np.sqrt(alpha)
    # Initial conditions

    v = f0(x)
    t = 0

    #xR = xR / np.sqrt(alpha)

    # Setting up Plot
    tplot = 0.5
    plotgap = int(round(tplot / dt))
    nplots = int(round(tmax / tplot))

    LX = len(x)
    data = np.zeros([nplots + 1, LX])
    data[0, :] = v
    tdata = np.zeros(nplots + 1)

    for i in range(1, nplots+1):

        for n in range(plotgap):
            t = t + dt
            # Euler
            v = euler(v, D1, D2, dt, alpha, L)

        data[i, :] = v
        if np.isnan(v).any():
            break
        # real time vector
        tdata[i] = t

    return x, tdata, data


xL = -60
xR = 60
#tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))
tmax = 100.0
dt = 0.01
alpha = 1.0

Ns = 2**np.arange(6, 10) + 1
Nmax = 513

hz = 2.0 * np.pi / Nmax
z = hz * np.arange(0, Nmax)
q = 2 * np.pi / (xR - xL)
z = z / q + xL
exact = exact_sol(alpha, z, np.arange(0, tmax + 0.5, 0.5))

fig, ax = plt.subplots(1, 2, figsize=(15, 10))

colors= ['m--', 'c--', 'b--', 'g--', 'r--']
legend = [r'$u(x)$']
ax[0].plot(z, exact[-1, :], 'b-')
for j in range(len(Ns)):
    legend = legend + ['N' + ' = ' + str(Ns[j])]
    X, tdata, aprox = Simulation(Ns[j], xL, xR, tmax, dt, alpha)

    T = len(tdata) - 1
    ut_hat = aprox[T, :]
    hx = 2.0 * np.pi / Ns[j]
    x = hx * np.arange(0, Ns[j])
    ut = fourier(z, ut_hat, Ns[j], x, hz)

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
    #name = 'Numerical_Solution_alpha=' + str(alpha) + '_T=100.png'
    plt.tight_layout()


N = 257
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

name = 'Exact_Solution' + '.png'
plt.tight_layout()
#plt.savefig(name)

plt.show()