import math
import numpy
from numpy import pi,cos,sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.linalg import toeplitz
from numpy import pi,arange,exp,sin,cos,zeros,tan,inf, dot, ones, sqrt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from numpy import pi,linspace,sin,exp,zeros,arange,real, ones, sqrt, array
from matplotlib.pyplot import figure, plot, show
from scipy.sparse import coo_matrix
from time import time
from Differentiation_Matrix import fourdif
from numba import jit
from scipy import optimize
from scipy.integrate import trapz
from matplotlib import rc
from matplotlib.ticker import LinearLocator, FormatStrFormatter

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

def f0(x):

    return np.exp(- 0.005 * x**2)

def f0_prime(x):

    return -0.01 * x * f0(x)

def F(z, x, t):

    return z + f0(z) * t - x


def exact_sol(x, t):
    exact = np.zeros([len(t), len(x)])
    exact[0, :] = f0(x)
    for i in range(1, len(t)):
        Z = np.zeros(len(x))
        for j in range(len(x)):
            zj = optimize.root(F, 0.0, args=(x[j], t[i]), tol=10 ** -10)
            Z[j] = zj.x

        u = f0(Z)
        exact[i, :] = u
    return exact

def plot(X, tdata, data):

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

xL = -60
xR = 60
tmax = - 1.0 / min(f0_prime(np.linspace(xL, xR, 100)))
dt = 0.001

N = 512

h = 2.0 * np.pi / N
x = h * np.arange(0, N)
q = 2 * np.pi / (xR - xL)
x = x / q + xL
tdata = np.arange(0, tmax + 0.5, 0.5)
exact = exact_sol(x, tdata)

plot(x, tdata, exact)