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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def f_interpolate(Z, t, exact_solution):

    f_x = interpolate.InterpolatedUnivariateSpline(Z, exact_solution[t, :])

    return f_x


def func(x, a, b):

    return a * np.exp(-b * x)
    #return np.log(a) + b * np.log(1/x)


def fourier(k, z, y_hat, h, N):
    F = np.zeros(len(z))
    for j in range(0, len(z)):

        F[j] = sum(y_hat * np.exp(1j * k * h * j)).real / N
    return F

def compare():
    alphas = [1.0, 0.5, 0.025, 0.01, 0.005]
    #alphas = [1.0]
    Nj = [2**j + 1 for j in range(4, 13)]

    xL = -60; xR = 60
    N = 8192
    h = 2.0 * np.pi / N

    Z = h * np.arange(0, N)
    p = 2 * np.pi / (xR - xL)
    Z = Z / p + xL

    dts = [0.01, 0.001, 0.0001, 0.00001]
    #dts = [0.00001]
    tdata = np.linspace(0, 100, 201)
    norm_dts=np.zeros([len(Nj), len(dts)])
    norm_t = np.array([norm_dts for j in range(4)])

    norma_alpha_L2_max = np.array([norm_dts for j in range(len(alphas))])
    norma_alpha_L2 = np.array([norm_t for j in range(len(alphas))])

    norma_alpha_max_max = np.array([norm_dts for j in range(len(alphas))])
    norma_alpha_max = np.array([norm_t for j in range(len(alphas))])

    colors = ['m-o', 'c-o', 'b-o', 'g-o', 'r-o']

    legend = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for alpha in alphas:
        legend = legend + [r'$\alpha$ = ' + str(alpha)]

        for dt in dts:

            maxnorm_L2 = []
            maxnorm_max = []
            norm_L2_t = np.zeros([4, len(Nj)])
            norm_max_t = np.zeros([4, len(Nj)])
            for j in range(0, len(Nj)):

                direction1 = 'Graphics/' + 'eps=' + str(alpha) + '/dt=' + str(dt) + '/N=' + str(Nj[j])
                element1 = '/norm_L2_t.npy'
                element2 = '/norm_max_t.npy'
                if not os.path.exists(direction1 + element1):
                    break
                norm_L2 = np.load(direction1 + element1)
                norm_max = np.load(direction1 + element2)

                maxnorm_L2.append(max(norm_L2))
                maxnorm_max.append(max(norm_max))
                norma_alpha_L2_max[alphas.index(alpha)][j, dts.index(dt)] = max(norm_L2)
                norma_alpha_max_max[alphas.index(alpha)][j, dts.index(dt)] = max(norm_max)
                for t in tdata[50:201:50]:
                    norm_L2_t[int(t / 25) - 1, j] = norm_L2[int(t)]
                    norm_max_t[int(t / 25) - 1, j] = norm_max[int(t)]
                    norma_alpha_L2[alphas.index(alpha)][int(t / 25) - 1][j, dts.index(dt)] = norm_L2[int(t)]
                    norma_alpha_max[alphas.index(alpha)][int(t / 25) - 1][j, dts.index(dt)] = norm_max[int(t)]

        Ns = 2 ** np.arange(4, 4 + len(maxnorm_L2))
        maxnorm_L2 = np.sort(maxnorm_L2)[::-1]
        maxnorm_max = np.sort(maxnorm_max)[::-1]
        #const1, pcov1 = curve_fit(func, Ns, maxnorm_L2, p0=(1.0, 0.001))
        #const2, pcov2 = curve_fit(func, Ns, maxnorm_max, p0=(1.0, 0.001))
        #const3, pcov3 = curve_fit(func, Ns, norm_L2_t[2, 0:len(maxnorm_L2)], p0=(1.0, 0.001))
        #const4, pcov4 = curve_fit(func, Ns, norm_max_t[2, 0:len(maxnorm_max)], p0=(1.0, 0.001))

        #s_1 = np.linspace(min(Ns), max(Ns), 1000)

        #print(const1)
        #fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax[0].set_yscale('log')
        # ax[0].plot(s_1, func(s_1, *const1), '--')
        ax[0].plot(Ns, maxnorm_L2, colors[alphas.index(alpha)])

        ax[1].set_yscale('log')
        ax[1].plot(Ns, maxnorm_max,colors[alphas.index(alpha)])
        # ax[1].plot(s_1, func(s_1, *const2))

        ax[0].set_xlabel(r'\textit{$N$}', fontsize=15)
        ax[0].set_ylabel(r'\textbf{$\displaystyle \max_{t \in [0, T]} \| u - u_N \|_{L^2}$}',
                         fontsize=20)

        ax[1].set_xlabel(r'\textit{$N$}', fontsize=15)
        ax[1].set_ylabel(r'\textbf{$\displaystyle \max_{t \in [0, T]} | u - u_N |$}',
                         fontsize=20)

        #          fontsize=16, color='black')

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        #ax.set_ylim(np.log(min(maxnorm_max)), np.log(max(maxnorm_L2)))
        ax[0].grid();
        ax[1].grid()

        params = {'legend.fontsize': 18,
                  'legend.handlelength': 0.1}
        plt.rcParams.update(params)

        ax[0].text(-0.1, 1.08, '(a)',
                   horizontalalignment='left', fontsize=20,
                   transform=ax[0].transAxes)
        ax[1].text(1.08, 1.08, '(b)',
                   horizontalalignment='left', fontsize=20,
                   transform=ax[0].transAxes)

        ax[0].legend(legend)
        ax[0].tick_params(axis='x', labelsize='15')
        ax[1].tick_params(axis='x', labelsize='15')
        ax[0].tick_params(axis='y', labelsize='15')
        ax[1].tick_params(axis='y', labelsize='15')

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
    
    ax[0].legend(legend)

    plt.tight_layout()
    name = 'alphas_Error_N.png'
    #plt.savefig(name)
    plt.show()
    '''''
    np.save('norma_alpha_L2_max', norma_alpha_L2_max)
    np.save('norma_alpha_L2', norma_alpha_L2)
    np.save('norma_alpha_max_max', norma_alpha_max_max)
    np.save('norma_alpha_max', norma_alpha_max)
            #const, pcov = curve_fit(func, Nj, norm_l2, p0=(1.0, 0.001))

        #plt.figure(2)
        #s = np.linspace(16, 4096, 1000)
        #plt.plot(Nj, norm_l2, '--')
        #plt.plot(s, func(s, *const))
    '''''

compare()

