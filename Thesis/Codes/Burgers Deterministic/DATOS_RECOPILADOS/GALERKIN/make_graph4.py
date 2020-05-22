import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import warnings
from scipy.fftpack import dct, idct
from scipy import interpolate
import matplotlib.tri as mtri
from scipy.interpolate import griddata
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.integrate import trapz
from numba import jit
from time import time

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def f_interpolate(Z, t, exact_solution):

    f_x = interpolate.InterpolatedUnivariateSpline(Z, exact_solution[t, :])

    return f_x


def func(x, a, b):

    return a * np.exp(-b * x)
    #return a / x**b

def fourier(k, z, y_hat, h, N):
    F = np.zeros(len(z))
    for j in range(0, len(z)):

        F[j] = sum(y_hat * np.exp(1j * k * h * j)).real / N
    return F

def compare():
    #alphas = [1.0, 0.5, 0.025, 0.01, 0.005]
    #alphas = [1.0, 0.5, 0.025]
    alphas=[0.005]
    #Nj = 2 ** np.arange(4, 12)
    Nj = [2 ** j for j in range(6, 11)]
    xL = -60; xR = 60
    N = 8192
    h = 2.0 * np.pi / N

    Z = h * np.arange(0, N)
    p = 2 * np.pi / (xR - xL)
    Z = Z / p + xL

    #dts = [0.01, 0.001, 0.0001, 0.00001]
    dt = 0.00001
    tdata = np.linspace(0, 100, 201)

    for alpha in alphas:
        direction1 = 'TEST/Exact_Solution/' + '/eps=' + str(alpha)
        element1 = '/sol_exact_' + str(alpha) + '_' + str(N)
        exact_solution = np.load(direction1 + element1 + '.npy')
        t=100.0
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        #ax[0].plot(Z,  exact_solution[2*int(t), :], 'bo')
        legend = [r'$u(x)$']
        s = 30
        ax[0].plot(Z[0::s], exact_solution[2*int(t), 0::s], 'k-', linewidth=3)
        colors= ['m--', 'c--', 'b--', 'g--', 'r--']

        for j in range(0, len(Nj)):

            direction1 = 'Graphics/' + 'eps=' + str(alpha) + '/dt=' + str(dt) + '/N=' + str(Nj[j])
            if not os.path.isdir(direction1 + '/aprox_t'):
                break

            element3 = '/aprox_t/aprox_t=' + str(t) + '.npy'
            #element4 = '/exact_t/exact_t=' + str(t) + '.npy'

            aprox = np.load(direction1 + element3)
            #exact = np.load(direction1 + element4)
            legend = legend + ['N' + ' = ' + str(Nj[j])]

            ax[0].plot(Z[0::s], aprox[0::s], colors[j], linewidth=1.5)
            ax[1].semilogy(Z[0::s], abs(aprox[0::s] - exact_solution[2*int(t), 0::s]), colors[j], linewidth=1.5)
            #ax[1].plot(Z, abs(aprox - exact))

            #plt.title('Continuity with respect to the initial conditions',
            #          fontsize=16, color='black')


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
         transform = ax[0].transAxes)
        ax[1].text(1.08, 1.08, '(b)',
                 horizontalalignment='left', fontsize=25,
         transform = ax[0].transAxes)

        ax[0].legend(legend)
        ax[0].tick_params(axis='x', labelsize='25')
        ax[1].tick_params(axis='x', labelsize='25')
        ax[0].tick_params(axis='y', labelsize='25')
        ax[1].tick_params(axis='y', labelsize='25')
        name = 'Numerical_Solution_alpha=' + str(alpha) + '_T=100.png'
        plt.tight_layout()
        plt.savefig(name)

        X1, Y1 = np.meshgrid(Z, tdata)
        W1 = exact_solution

        # ax2 = fig2.gca(projection='3d')
        fig1 = plt.figure(figsize=(15, 15))
        ax1 = fig1.gca(projection='3d')

        surf1 = ax1.plot_surface(X1, Y1, W1, cmap=cm.Spectral_r, rstride=1, cstride=1)

        # ax2.contour3D(X, Y, W, cmap='binary')

        ax1.set_zlim(0, 1.0)
        ax1.set_xlabel(r'$x$', fontsize=25, labelpad=20)
        ax1.set_ylabel(r'$t$', fontsize=25, labelpad=20)
        ax1.set_zlabel(r'$u(x, t)$', fontsize=25, labelpad=30)
        ax1.view_init(30, -80)

        ax1.tick_params(axis='x', labelsize='25', pad=5)
        ax1.tick_params(axis='y', labelsize='25', pad=5)
        ax1.tick_params(axis='z', labelsize='25', pad=12)

        name = 'Exact_Solution_alpha=' + str(alpha) + '.png'
        plt.tight_layout()
        plt.savefig(name)

        #plt.show()

        # ax2.zaxis._axinfo['juggled'] = (2, 1, 2)

        fig2 = plt.figure(figsize=(15, 15))

        Nk = 2**11

        hx = 2.0 * np.pi / 201
        x = hx * np.arange(0, 201)
        p = 2 * np.pi / (xR - xL)
        x = x / p + xL

        k = np.zeros(Nk)
        k[0: int(Nk / 2)] = np.arange(0, int(Nk / 2))
        k[int(Nk / 2):] = np.arange(-int(Nk / 2), 0, 1)

        direction2 = 'TEST/Simulation_Data/eps=' + str(alpha) + '/N=' + str(Nk) + '/dt=' + str(dt)
        element_2 = '/alpha=' + str(alpha) + '_N=' + str(Nk) + '_dt=' + str(dt)

        coeff = np.load(direction2 + element_2 + '_data.npy')
        aprox_full = np.zeros([len(tdata), 201])
        indx = ['(a)', '(b)']
        for t in range(0, len(tdata)):
            aprox_full[t, :] = fourier(k, x, coeff[t, :], hx, Nk)

        X, Y = np.meshgrid(x, tdata)
        W = aprox_full

        #ax2 = fig2.gca(projection='3d')
        ax2 = fig2.gca(projection='3d')

        surf = ax2.plot_surface(X, Y, W, cmap=cm.Spectral_r, rstride=1, cstride=1)

        #ax2.contour3D(X, Y, W, cmap='binary')

        #ax2.set_title(indx[Nj[5:7].index(Nk)], loc='left', fontsize=15)
        #ax2.zaxis._axinfo['juggled'] = (1, 2, 2)

        ax2.set_zlim(0, 1.0)
        ax2.set_xlabel(r'$x$', fontsize=25, labelpad=20)
        ax2.set_ylabel(r'$t$', fontsize=25, labelpad=20)
        ax2.set_zlabel(r'$u(x, t)$', fontsize=25, labelpad=30)
        ax2.view_init(30, -80)

        ax2.tick_params(axis='x', labelsize='25', pad=5)
        ax2.tick_params(axis='y', labelsize='25', pad=5)
        ax2.tick_params(axis='z', labelsize='25', pad=12)

        name = 'Numerical_Solution_alpha=' + str(alpha) + '.png'
        plt.tight_layout()
        plt.savefig(name)
        #fig2.colorbar(surf, shrink=0.5, aspect=5)
        #plt.show()

                #const, pcov = curve_fit(func, Nj, norm_l2, p0=(1.0, 0.001))

            #plt.figure(2)
            #s = np.linspace(16, 4096, 1000)
            #plt.plot(Nj, norm_l2, '--')
            #plt.plot(s, func(s, *const))


compare()

#plt.show()