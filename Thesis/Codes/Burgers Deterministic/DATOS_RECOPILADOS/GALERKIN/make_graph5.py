import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import warnings
from scipy.fftpack import dct, idct
from scipy import interpolate
from tabulate import tabulate

from scipy.interpolate import griddata
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.integrate import trapz


from matplotlib import rc
import matplotlib.ticker as ticker

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def f_interpolate(Z, t, exact_solution):
    f_x = interpolate.InterpolatedUnivariateSpline(Z, exact_solution[t, :])

    return f_x


def func(x, a, b):
    return a * np.exp(-b * x)
    # return np.log(a) + b * np.log(1/x)


def fourier(k, z, y_hat, h, N):
    F = np.zeros(len(z))
    for j in range(0, len(z)):
        F[j] = sum(y_hat * np.exp(1j * k * h * j)).real / N
    return F


def compare():
    alphas = [1.0, 0.5, 0.025, 0.01, 0.005]
    # alphas = [1.0]
    Nj = [2 ** j for j in range(8, 12)]
    #Nj = [4096]
    xL = -60;
    xR = 60
    N = 8192
    h = 2.0 * np.pi / N

    Z = h * np.arange(0, N)
    p = 2 * np.pi / (xR - xL)
    Z = Z / p + xL

    dts = 0.00001
    # dts = [0.00001]
    tdata = np.linspace(0, 100, 201)

    #colors = ['m-o', 'c-o', 'b-o', 'g-o', 'r-o']


    for alpha in alphas:
        legend = []
        fig, ax = plt.subplots(figsize=(10, 5))

        for j in range(0, len(Nj)):
            legend = legend + [r'$N$ = ' + str(Nj[j])]
            direction1 = 'Graphics/' + 'eps=' + str(alpha) + '/dt=' + str(dts) + '/N=' + str(Nj[j])
            element1 = '/norm_L2.npy'
            element2 = '/norm_max.npy'
            if not os.path.isdir(direction1 + '/aprox_t'):
                break
            norm_L2 = np.load(direction1 + element1)

            # print(const1)
            # fig2, ax2 = plt.subplots(figsize=(10, 10))

            # ax[0].plot(s_1, func(s_1, *const1), '--')
            #ax.plot(tdata, norm_L2, colors[alphas.index(alpha)])

            ax.plot(tdata, norm_L2, '*-')
            ax.set_xticks(np.linspace(0, 100, 11))
            ax.set_xlabel(r'\textit{$N$}', fontsize=15)
            ax.set_ylabel(r'\textbf{$\displaystyle \| u - u_N \|_{L^2}$}',
                             fontsize=20)

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            # ax.set_ylim(np.log(min(maxnorm_max)), np.log(max(maxnorm_L2)))
            ax.grid()

            params = {'legend.fontsize': 12, 'legend.handlelength': 0.1}
            plt.rcParams.update(params)

            #ax.text(-0.1, 1.08, '(a)', horizontalalignment='left', fontsize=20, transform=ax[0].transAxes)

            #ax[0].legend(legend)
            ax.tick_params(axis='x', labelsize='15')
            ax.tick_params(axis='x', labelsize='15')

            ax.legend(legend)

            #ax.ticklabel_format(axis='y', style='sci', useMathText=None)
            y_labels = ax.get_yticks()
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0e'))

            plt.tight_layout()
            name = 'alphas_Error_N.png'
            # plt.savefig(name)
        plt.show()

        '''''
        ax[0].set_yscale('log')
        #ax[0].plot(s_1, func(s_1, *const1), '--')
        ax[0].plot(Ns, maxnorm_L2, color='blue', marker='o')

        ax[1].set_yscale('log')
        ax[1].plot(Ns, maxnorm_max, color='blue', marker='o')
        #ax[1].plot(s_1, func(s_1, *const2))

        ax[0].set_xlabel(r'\textit{$N$}', fontsize=15)
        ax[0].set_ylabel(r'\textbf{$\| u - u_N \|_{L^2}$}',
                   fontsize=20)

        ax[1].set_xlabel(r'\textit{$N$}', fontsize=15)
        ax[1].set_ylabel(r'\textbf{$\| u - u_N \|_{\infty}$}',
                   fontsize=20)
        #ax[0].set_title('Continuity with respect to the initial conditions',
        #          fontsize=16, color='black')

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax[0].set_ylim(np.log(min(maxnorm_max)), np.log(max(maxnorm_L2)))
        ax[0].grid(); ax[1].grid()

        ''''''
        #each time
        fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))

        for t in range(1, len(norm_L2_t), 2):
            ax1[0].set_yscale('log')
            ax1[0].plot(Ns, norm_L2_t[t, 0:len(maxnorm_L2)], color='blue', marker='o')

            ax1[1].set_yscale('log')
            #ax1[0].plot(s_1, func(s_1, *const3))
            ax1[1].plot(Ns, norm_max_t[t, 0:len(maxnorm_max)], color='blue', marker='o')
            #ax1[1].plot(s_1, func(s_1, *const4))

            ax1[0].set_xlabel(r'\textit{$N$}', fontsize=15)
            ax1[0].set_ylabel(r'\textbf{$\| u - u_N \|_{L^2}$}',
                             fontsize=20)

            ax1[1].set_xlabel(r'\textit{$N$}', fontsize=15)
            ax1[1].set_ylabel(r'\textbf{$\| u - u_N \|_{\infty}$}',
                             fontsize=20)
            # ax[0].set_title('Continuity with respect to the initial conditions',
            #          fontsize=16, color='black')

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            ax1[0].grid(); ax1[1].grid()
        '''''
    # np.save('norma_alpha_L2_max', norma_alpha_L2_max)
    # np.save('norma_alpha_L2', norma_alpha_L2)
    # np.save('norma_alpha_max_max', norma_alpha_max_max)
    # np.save('norma_alpha_max', norma_alpha_max)
    # const, pcov = curve_fit(func, Nj, norm_l2, p0=(1.0, 0.001))

    # plt.figure(2)
    # s = np.linspace(16, 4096, 1000)
    # plt.plot(Nj, norm_l2, '--')
    # plt.plot(s, func(s, *const))


compare()

