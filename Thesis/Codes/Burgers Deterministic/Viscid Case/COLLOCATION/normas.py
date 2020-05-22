import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import warnings
from scipy.fftpack import dct, idct
from scipy import interpolate

from scipy.interpolate import griddata
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.integrate import trapz
from Galerkin import Simulation, fourier, f0_prime
from Exact_inviscid import exact_sol

from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def graph(Ns, dt):

    fig1, ax1 = plt.subplots(figsize=(5, 5))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = ['m--', 'c--', 'b--', 'g--', 'r--']
    legend1 = [r'$u(x)$'] + [r'$\alpha = 1.0$'] + [r'$\alpha = 1.0 \times 10^{-1}$'] + [r'$\alpha = 1.0 \times 10^{-5}$']
    ax1.plot(z, exact[-1, :], 'b-')

    X, tdata, aprox1 = Simulation(Ns[-1], xL, xR, tmax, dt, alphas[0])
    X, tdata, aprox2 = Simulation(Ns[-1], xL, xR, tmax, dt, alphas[1])
    X, tdata, aprox3 = Simulation(Ns[-1], xL, xR, tmax, dt, alphas[2])

    ax1.plot(X, aprox1[-1, :], colors[0], linewidth=1.5)
    ax1.plot(X, aprox2[-1, :], colors[1], linewidth=1.5)
    ax1.plot(X, aprox3[-1, :], colors[4], linewidth=1.5)
    ax1.grid()

    ax1.set_xlim(xL, xR)
    ax1.set_xticks(np.linspace(xL, xR, 7))
    ax1.tick_params(axis='x', labelsize='25')
    ax1.tick_params(axis='y', labelsize='25')

    params = {'legend.fontsize': 25,
              'legend.handlelength': 0.5}
    plt.rcParams.update(params)
    ax1.legend(legend1)
    ax1.set_xlabel(r'$x$', fontsize=25, color='black')
    ax1.set_ylabel(r'\textbf{$u_N$}',
                     fontsize=25, color='black')


    plt.tight_layout()
    plt.savefig('varios_alphas.png')

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    legend = [r'$u(x)$']
    ax[0].plot(z, exact[-1, :], 'b-')
    for j in range(len(Ns)):
        legend = legend + ['N' + ' = ' + str(Ns[j])]
        X, tdata, aprox = Simulation(Ns[j], xL, xR, tmax, dt, alphas[2])
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
    plt.show()

def graph3d():
    x, tdata, Z = Simulation(N[-1], xL, xR, tmax, dts[1], alphas[-1])
    X, Y = np.meshgrid(x, tdata)
    fig1 = plt.figure(figsize=(15, 15))
    ax1 = fig1.gca(projection='3d')

    surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.Spectral_r, rstride=1, cstride=1)

    # ax2.contour3D(X, Y, W, cmap='binary')

    ax1.set_zlim(0, 1.0)
    ax1.set_xlabel(r'$x$', fontsize=25, labelpad=20)
    ax1.set_ylabel(r'$t$', fontsize=25, labelpad=20)
    ax1.set_zlabel(r'$u(x, t)$', fontsize=25, labelpad=30)
    ax1.view_init(30, -80)

    ax1.tick_params(axis='x', labelsize='20', pad=5)
    ax1.tick_params(axis='y', labelsize='20', pad=5)
    ax1.tick_params(axis='z', labelsize='20', pad=12)

    name = 'Numerical_Solution_Inviscid' + '.png'
    plt.tight_layout()
    #plt.savefig(name)

    plt.show()


xL = -60
xR = 60
tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 3000)))
dts = [0.01, 0.001, 0.0001]

alphas = [1.0, 0.1, 0.00001]

N = 2**np.arange(4, 9)
M = 512


hz = 2.0 * np.pi / M
z = hz * np.arange(0, M)
q = 2 * np.pi / (xR - xL)
z = z / q + xL
t = np.arange(0, tmax + 0.5, 0.5)

max_L2 = np.zeros([len(N), len(dts)])
max_max = np.zeros([len(N), len(dts)])

exact = exact_sol(z, t)
graph(N, dts[1])
#graph3d()

'''''
for p in range(len(dts)):
    for j in range(len(N)):
        Nj = N[j]
        k = np.zeros(Nj)
        k[0: int(Nj / 2)] = np.arange(0, int(Nj / 2))
        k[int(Nj / 2) + 1:] = np.arange(-int(Nj / 2) + 1, 0, 1)

        x, tdata, aprox = Simulation(Nj, xL, xR, tmax, dts[p], alphas[-1])
        norm_L2_t = np.zeros(len(t))
        norm_max_t = np.zeros(len(t))
        for i in range(len(tdata)):
            coeff = np.fft.fft(aprox[i, :])
            Exact_t = exact[i, :]
            aprox_t = fourier(k, z, coeff, hz, Nj)

            E = Exact_t - aprox_t
            norm_L2_t[i] = np.sqrt(trapz(E**2, z))
            norm_max_t[i] = max(abs(E))
        max_L2[j, p] = max(norm_L2_t)
        max_max[j, p] = max(norm_max_t)

np.save('norm_L2', max_L2)
np.save('norm_max', max_max)

print(max_L2)
print(max_max)
'''''