import numpy as np

import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import LineCollection

from ExactSol_Burgers import exact
from Fourier_Galerkin import Simulation


def convergence():
    N = 128
    xL = -5.0
    xR = 5.0
    tmax = 10.0

    sol_exact = np.loadtxt('Graphics/' + 'sol_0.1')

    plt.figure()
    for Nj in 2**np.arange(1, 3):
        tdata, data, X = Simulation(Nj, xL, xR, tmax)
        Error = max(abs(data[0:666, 4] - sol_exact[:, 4]))

        plt.plot(N, Error)


convergence()


#exact_solution()


plt.show()