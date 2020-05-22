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

from Fourier_Collocation import Simulation


def solver(z, N, xL, xR, tmax, dt):

    tdata, x, data = Simulation(N, xL, xR, tmax, dt)
    solutions = np.zeros([len(tdata), len(z)])
    ns = np.arange(-int(N/2), int(N/2) + 1, 1)
    un = np.zeros(N, dtype='complex')

    for k in range(len(tdata)):
        ut = data[k, :]
        for j in range(0, N):
            y = ut * np.exp(-1j * ns[j] * x) / (2.0 * np.pi)
            un[j] = trapz(y, x)

        for i in range(0, len(z)):
            if np.allclose(i * 2.0 * np.pi / N, x[i]):
                solutions[i] = ut[i]
            else:
                solutions[i] = sum(un * np.exp(1j * ns * i * 2.0 * np.pi / N)).real
    return solutions


# plot graphics
def plot(tdata, x, data):

    if np.isnan(data).any() == False:
        X, Y = np.meshgrid(X, tdata)
        Z = data
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.1)

        ax.set_zlim(0, 1.0)
        ax.set_xlabel(r'$x$', fontsize=15)
        ax.set_ylabel(r'$t$', fontsize=15)
        ax.set_zlabel(r'$u$', fontsize=15)

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

N = 257
xL = -60.0
xR = 60.0
tmax = 100.0
dt = 0.01
# Diffusivity constant
alpha = 0.01

plot(solver(z, u0, N, ))
