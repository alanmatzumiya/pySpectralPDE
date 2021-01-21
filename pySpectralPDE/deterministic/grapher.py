import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


class graph_defaults:

    @staticmethod
    def set_graph(ax, legend, type_graph):
        if type_graph == 'pointwise_max':
            ax[0].set_xlabel(r'\textit{$N$}', fontsize=15)
            ax[0].set_ylabel(r'\textbf{$\displaystyle \max_{t \in [0, T]} \| u - u_N \|_{L^2}$}',
                             fontsize=20)
            ax[1].set_xlabel(r'\textit{$N$}', fontsize=15)
            ax[1].set_ylabel(r'\textbf{$\displaystyle \max_{t \in [0, T]} | u - u_N |$}',
                             fontsize=20)
        else:
            ax[0].set_xlabel(r'$x$', fontsize=25, color='black')
            ax[0].set_ylabel(r'\textbf{$u_N$}',
                             fontsize=25, color='black')
            ax[1].set_xlabel(r'$x$', fontsize=25, color='black')
            ax[1].set_ylabel(r'\textbf{$| u - u_N |$}',
                             fontsize=25, color='black')
        ax[0].grid()
        ax[1].grid()
        rc_params = {'legend.fontsize': 25,
                     'legend.handlelength': 0.5}
        plt.rcParams.update(rc_params)
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

    def error_max(self, params, error, name='error_', *args):
        nu = params['nu']
        N = params['N']
        L2_distance = np.max(error['L2'])
        max_distance = np.max(error['L2'])

        colors = ['m-o', 'c-o', 'b-o', 'g-o', 'r-o']
        legend = [r'$\alpha$ = ' + str(nu[0])]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_yscale('log')
        ax[0].plot(N, L2_distance, colors[0])
        ax[1].set_yscale('log')
        ax[1].plot(N, max_distance, colors[0])
        self.set_graph(ax, legend, 'pointwise_max')
        plt.tight_layout()
        plt.savefig(name+'nu='+str(nu)+'.png')

    def error_time(self, params, exact, time, name='error_', *args):
        nu = params['nu']
        N = params['N']
        xL = params['xL']
        xR = params['xR']
        data = params['data']
        tdata = params['tdata']
        T = tdata.index(time)

        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].set_xlim(xL, xR)
        ax[1].set_xlim(xL, xR)
        ax[0].set_xticks(np.linspace(xL, xR, 7))
        ax[1].set_xticks(np.linspace(xL, xR, 7))
        legend = [r'$u(x)$']
        colors = ['m--', 'c--', 'b--', 'g--', 'r--']
        legend = legend + ['N' + ' = ' + str(N)]
        ax[0].plot(data[T, :], exact[T, :], colors[0], linewidth=1.5)
        ax[1].semilogy(data[T, :], abs(data[T, :] - exact[T, :]), colors[0], linewidth=1.5)
        self.set_graph(ax, legend, 'pointwise_T')
        plt.tight_layout()
        plt.savefig(name+'nu='+str(nu)+'_T='+str(time)+'.png')

    def graph_3d(self, x, t, data, name='exact3d'):

        # Setting up Plot
        X, Y = np.meshgrid(x, t)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, data, cmap='Spectral_r', rstride=1, cstride=1)

        ax.set_zlim(np.min(data), np.max(data))
        ax.set_xlabel(r'$x$', fontsize=25, labelpad=20)
        ax.set_ylabel(r'$t$', fontsize=25, labelpad=20)
        ax.set_zlabel(r'$u(x, t)$', fontsize=25, labelpad=30)
        ax.view_init(30, -80)
        ax.tick_params(axis='x', labelsize='25', pad=5)
        ax.tick_params(axis='y', labelsize='25', pad=5)
        ax.tick_params(axis='z', labelsize='25', pad=12)

        plt.tight_layout()
        plt.savefig(name+'.png')


    def plot_diffusion(self, u_analytical, u, x, NT, TITLE):
        """
        Plots the 1D velocity field
        """

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        plt.figure()
        ax = plt.subplot(111)
        colour = iter(cm.rainbow(np.linspace(0, 20, NT)))
        for n in range(0, NT, 20):
            c = next(colour)
            ax.plot(x, u[:, n], 'ko', markerfacecolor='none', alpha=0.5, label='i=' + str(n) + ' numerical')
            ax.plot(x, u_analytical[:, n], linestyle='-', c=c, label='i=' + str(n) + ' analytical')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2)
        plt.xlabel('x (radians)')
        plt.ylabel('u (m/s)')
        plt.ylim([0, 8.0])
        plt.xlim([0, 2.0 * PI])
        plt.title(TITLE)
        plt.show()
