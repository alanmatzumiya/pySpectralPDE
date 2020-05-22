import numpy as np

import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib import cm
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def Graph(xSpace, tim, simulation1, simulation2):
    """
    This function calculates the integral of the norm between two solutions
    with different initial conditions for a fixed point in real space

    Parameters
    ----------
    xSpace : array; discretized real space
    tim : array; discretized time
    simulation1 : array, shape(len(tim), len(xSpace))
        Array containing the solutions of partial equation
    simulation2 : array, shape(len(tim), len(xSpace))
        Array containing the solutions of partial equation with the IC approximated

    Returns
    -------
    Graphs : object,
        Graphs of the simulations

    """
    X, Y = np.meshgrid(xSpace, tim)

    # Graph simulation 1
    fig1 = pl.figure(1)
    ax1 = Axes3D(fig1)
    Z = simulation1

    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax1.set_xlabel('x', fontsize=20)
    ax1.set_ylabel(r'\textit{time} (t)', fontsize=20)
    ax1.set_zlabel(r'\textit{U} (t, x)', fontsize=20)

    # Graph simulation 2
    fig2 = pl.figure(2)
    ax2 = Axes3D(fig2)
    W = simulation2

    ax2.plot_surface(X, Y, W, cmap=cm.coolwarm)
    ax2.set_xlabel('x', fontsize=20)
    ax2.set_ylabel(r'\textit{time} (t)', fontsize=20)
    ax2.set_zlabel(r'\textit{U} (t, x)', fontsize=20)

    fig1.savefig('Graphics/simulation_IC')
    fig2.savefig('Graphics/simulation_Approximation')

    # Graph inicial condition - approximation
    for i in range(0, 11, 2):

        fig3, ax3 = plt.subplots()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.xlabel(r'\textit{x}', fontsize=15)
        plt.ylabel(r'\textit{U} (x)', fontsize=15)
        plt.title('Approximation with chebycheb polynomials',
                  fontsize=16, color='black')
        u = list(tim).index(i)
        ax3.grid()
        ax3.plot(xSpace,simulation1[u, :], 'r', label = 'Approximation')
        ax3.plot(xSpace, simulation2[u, :], 'b', label = r'\textit{U} (x)')
        ax3.legend()
        name = 'Approximation_t=' + str(i)
        fig3.savefig('Graphics/' + name)
