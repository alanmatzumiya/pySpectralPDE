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

# from ExactSol_Burgers import exact
# from Fourier_Galerkin import Simulation


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
    alphas = [1.0, 0.5, 0.025, 0.01, 0.005]
    #alphas = [1.0]
    Nj = 2 ** np.arange(4, 13)

    xL = -60; xR = 60
    N = 8192
    h = 2.0 * np.pi / N

    Z = h * np.arange(0, N)
    p = 2 * np.pi / (xR - xL)
    Z = Z / p + xL

    dts = [0.01, 0.001, 0.0001, 0.00001]
    #dts = [0.00001]
    tdata = np.linspace(0, 100, 201)

    for alpha in alphas:
        direction1 = 'TEST/Exact_Solution/' + '/eps=' + str(alpha)
        element1 = '/sol_exact_' + str(alpha) + '_' + str(N)

        exact_solution = np.load(direction1 + element1 + '.npy')

        for dt in dts:

            for j in Nj:
                k = np.zeros(j)
                k[0: int(j / 2)] = np.arange(0, int(j / 2))
                k[int(j / 2):] = np.arange(-int(j / 2), 0, 1)

                direction2 = 'TEST/Simulation_Data/eps=' + str(alpha) + '/N=' + str(j) + '/dt=' + str(dt)
                element_2 = '/alpha=' + str(alpha) + '_N=' + str(j) + '_dt=' + str(dt)

                direction3 = 'Graphics/eps=' + str(alpha) + '/dt=' + str(dt) + '/N=' + str(j)
                if not os.path.isdir(direction2):
                    break

                if not os.path.isdir(direction3 + '/aprox_t'):
                    os.mkdir(direction3 + '/aprox_t')
                    os.mkdir(direction3 + '/exact_t')

                coeff = np.load(direction2 + element_2 + '_data.npy')

                '''''
                for t in range(0, len(tdata), 20):
                    Exact_t = exact_solution[t, :]
                    aprox_t = fourier(k, Z, coeff[t, :], h, j)

                    np.save(direction3 + '/aprox_t/aprox_t=' + str(tdata[t]), aprox_t)
                    np.save(direction3 + '/exact_t/exact_t=' + str(tdata[t]), Exact_t)
                '''''

            #const, pcov = curve_fit(func, Nj, norm_l2, p0=(1.0, 0.001))

        #plt.figure(2)
        #s = np.linspace(16, 4096, 1000)
        #plt.plot(Nj, norm_l2, '--')
        #plt.plot(s, func(s, *const))


compare()

plt.show()