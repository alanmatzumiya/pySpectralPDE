import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib import cm

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


class graph_defaults:

    def __init__(self, x, t, data):
        self.x = x
        self.t = t
        self.data = data

    def graph_distance(self, distance, name='stability'):
        """
        This function constructs the graphs of the simulations
        and the calculation of the convergence
        Returns
        -------
        graph : object
        """

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.plot(self.t, distance)
        plt.xlim(self.t[0], self.t[len(self.t) - 1])
        plt.xlabel(r'\textit{time} (t)', fontsize=15)
        plt.ylabel(r'\textbf{$\| \Psi_{t}^{x} - \Psi_{t}^{y} \|_{\mathcal{L}(\mathcal{H}, \mu)^{2}}^{2}$}', fontsize=20)
        plt.title('Continuity with respect to the initial conditions',
                  fontsize=16, color='black')
        ax.grid()
        fig.savefig(name)

    def graph_time(self, times, data_approx, name='Approximation_t='):

        for i in times:
            fig, ax = plt.subplots()
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            plt.xlabel(r'\textit{x}', fontsize=15)
            plt.ylabel(r'\textit{U} (x)', fontsize=15)
            plt.title('Approximation with chebycheb polynomials',
                      fontsize=16, color='black')
            ax.grid()
            ax.plot(self.x, self.data[self.t.index(i), :], 'r', label='Approximation')
            ax.plot(self.x, data_approx[self.t.index(i), :], 'b', label=r'\textit{U} (x)')
            ax.legend()
            fig.savefig(name + str(i))

    def graph_3d(self, name='solution'):
        """
        This function calculates the integral of the norm between two solutions
        with different initial conditions for a fixed point in real space
        Parameters
        ----------
        xSpace : array; discretized real space
        tim : array; discretized time
        data : array, shape(len(tim), len(xSpace))
            Array containing the solutions of partial equation

        Returns
        -------
        Graphs : object,
            Graphs of the simulations
        """
        X, Y = np.meshgrid(self.x, self.t)

        fig1 = pl.figure(1)
        ax1 = Axes3D(fig1)
        ax1.plot_surface(X, Y, self.data, cmap='coolwarm')
        ax1.set_xlabel('x', fontsize=20)
        ax1.set_ylabel(r'\textit{time} (t)', fontsize=20)
        ax1.set_zlabel(r'\textit{U} (t, x)', fontsize=20)

        fig1.savefig(name)
