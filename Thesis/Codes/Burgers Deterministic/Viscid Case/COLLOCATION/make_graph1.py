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


def fourier(z, ut, N, x, h):
    F = np.zeros(len(z))

    ns = np.arange(-int(N/2), int(N/2) + 1, 1)

    un = np.zeros(N, dtype='complex')


    for j in range(0, N):
        y = ut * np.exp(-1j * ns[j] * x) / (2.0 * np.pi)
        un[j] = trapz(y, x)

    for i in range(0, len(z)):
        if np.allclose(h * i, z[i]):
            F[i] = ut[i]
        else:
            F[i] = sum(un * np.exp(1j * ns * h * i)).real
    return F

def compare():
    alphas = [1.0, 0.5, 0.025, 0.01, 0.005]
    #alphas = [0.025]
    Nj = 2 ** np.arange(4, 13) + 1
    #Nj= [65]

    xL = -60; xR = 60
    N = 8192
    h = 2.0 * np.pi / N

    z = h * np.arange(0, N)
    p = 2 * np.pi / (xR - xL)
    Z = z / p + xL

    dts = [0.01, 0.001, 0.0001, 0.00001]
    #dts = [0.0001]
    tdata = np.linspace(0, 100, 201)

    for alpha in alphas:
        direction1 = 'Generated_Data/Exact_Solution/' + '/eps=' + str(alpha)
        element1 = '/sol_exact_' + str(alpha) + '_' + str(N)

        exact_solution = np.load(direction1 + element1 + '.npy')

        for dt in dts:
            plt.figure()
            for j in Nj:
                norm_max = []
                norm_L2 = []
                norm_max_max = []
                norm_L2_max = []

                hx = 2.0 * np.pi / j
                x = hx * np.arange(0, j)

                direction2 = 'Generated_Data/Simulation_Data/eps=' + str(alpha) + '/N=' + str(j) + '/dt=' + str(dt)
                element_2 = '/alpha=' + str(alpha) + '_N=' + str(j) + '_dt=' + str(dt)

                direction3 = 'Graphics/eps=' + str(alpha) + '/dt=' + str(dt) + '/N=' + str(j)
                if not os.path.isdir(direction2):
                    break

                aprox = np.load(direction2 + element_2 + '_data.npy')


                for t in range(0, len(tdata)):
                    Exact_t = exact_solution[t, :]

                    aprox_t = fourier(Z, aprox[t, :], j, x, h)

                    E = abs(Exact_t - aprox_t)**2
                    L2_t = np.sqrt(trapz(E, Z))
                    max_t = max(abs(aprox_t - Exact_t))

                    norm_L2.append(L2_t)
                    norm_max.append(max_t)

                #np.save(direction3 + '/exact_t=100', Exact_t)
                #np.save(direction3 + '/aprox_t=100', aprox_t)
                #norm_L2_max.append(max(norm_L2))
                #norm_max_max.append(max(norm_max))
                np.save(direction3 + '/norm_L2_t', norm_L2)
                np.save(direction3 + '/norm_max_t', norm_max)

                #np.save(direction3 + '/norm_L2_max', norm_L2_max)
                #np.save(direction3 + '/norm_max_max', norm_max_max)


            #const, pcov = curve_fit(func, Nj, norm_l2, p0=(1.0, 0.001))

        #plt.figure(2)
        #s = np.linspace(16, 4096, 1000)
        #plt.plot(Nj, norm_l2, '--')
        #plt.plot(s, func(s, *const))


compare()
