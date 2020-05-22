import numpy as np
import matplotlib.pyplot as plt

import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.ticker as mticker

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

def graph3d(x, tdata, Z, W):

    X, Y = np.meshgrid(x, tdata)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig1 = plt.figure(figsize=(15, 15))
    ax1 = fig1.gca(projection='3d')

    surf = ax1.plot_surface(X, Y, Z, cmap=cm.cividis, rstride=1, cstride=1)
    ax1.plot_surface(X, Y, W, cmap=cm.Reds, rstride=1, cstride=1)
    # ax2.contour3D(X, Y, W, cmap='binary')

    ax1.set_zlim(0, 1.0)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 10.0)
    ax1.set_xlabel(r'$x$', fontsize=25, labelpad=20)
    ax1.set_ylabel(r'$t$', fontsize=25, labelpad=20)
    ax1.set_zlabel(r'$U(x, t)$', fontsize=25, labelpad=30)
    ax1.view_init(20, -70)


    ax1.set_xticks(np.arange(0, 1 + 0.1, 0.1))
    ax1.set_yticks(np.arange(0, 10 + 1, 1))
    ax1.set_zticks(np.arange(0, 1 + 0.1, 0.1))
    ax1.tick_params(axis='x', labelsize='20', pad=5)
    ax1.tick_params(axis='y', labelsize='20', pad=5)
    ax1.tick_params(axis='z', labelsize='20', pad=12)


    # Add a color bar which maps values to colors.
    cbar = fig1.colorbar(surf, shrink=0.75, aspect=15)
    cbar.ax.tick_params(labelsize=20)
    name = 'Numerical_Solution_Stochastic' + '.png'
    plt.tight_layout()
    plt.savefig(name)

    plt.show()

def graph(z, x, y):

    fig1, ax1 = plt.subplots(figsize=(10, 10))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = ['m--', 'c--', 'b--', 'g--', 'r--']
    legend1 = [r'$x(\xi)$'] + [r'$y(\xi)$']
    ax1.plot(z, x, 'b-')

    ax1.plot(z, y, colors[-1], linewidth=1.5)
    ax1.grid()

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(np.arange(0, 1 + 0.1, 0.1))
    ax1.set_yticks(np.arange(0, 1 + 0.1, 0.1))
    ax1.tick_params(axis='x', labelsize='20')
    ax1.tick_params(axis='y', labelsize='20')

    params = {'legend.fontsize': 25,
              'legend.handlelength': 0.5}
    plt.rcParams.update(params)
    ax1.legend(legend1)
    ax1.set_xlabel(r'$\xi$', fontsize=25, color='black')
    ax1.set_ylabel(r'\textbf{$X(\xi, 0)$}',
                     fontsize=25, color='black')


    plt.tight_layout()
    plt.savefig('IC')
    plt.show()

def Norma(t, norms):

    fig1, ax1 = plt.subplots(figsize=(10, 10))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = ['m--', 'c--', 'b--', 'g--', 'r--']

    ax1.semilogy(t, norms, colors[-1], linewidth=1.5)
    ax1.grid()

    ax1.set_xlim(0, 10)
    #ax1.set_ylim(0, 1)
    ax1.set_xticks(np.arange(0, 10 + 1, 1))
    #ax1.set_yticks(np.arange(0, 1 + 0.1, 0.1))
    ax1.tick_params(axis='x', labelsize='20')
    ax1.tick_params(axis='y', labelsize='20')

    ax1.set_xlabel(r'\textit{time} (t)', fontsize=25)
    ax1.set_ylabel(r'\textbf{$\| \Psi_{t}^{x} - \Psi_{t}^{y} \|_{\left(L^2(\mathcal{H}, \mu) \right)^2}$}', fontsize=25)

    #f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    #g = lambda x, pos: "${}$".format(f._formatSciNotation('%10e' % x))
    #plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))

    plt.tight_layout()
    plt.savefig('norms')
    plt.show()


z = np.load('xSpace.npy')
z= z[0::8]

tdata = np.load('times.npy')

tdata = tdata[0::9]
Z = np.load('data_1.npy')
W = np.load('data_2.npy')
x = Z[0, 0::8]
y = W[0, 0::8]

norm = np.load('norms.npy')
norm = norm[0::9]

Norma(tdata, norm)
