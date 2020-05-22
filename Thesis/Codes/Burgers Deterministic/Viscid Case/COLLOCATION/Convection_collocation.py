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


def f0(x):

    return np.exp(- 0.005 * x**2)

def F(z, x, t):

    return x - f0(z) * t - z

@jit
def euler(v, D1, D2, dt, alpha, L):
    v = v + dt * (- 0.5 * L * np.dot(D1, v**2))

    return v



def Simulation(N, xL, xR, tmax, dt, alpha):
    # Grid
    x, D1 = fourdif(N, 1)
    x, D2 = fourdif(N, 2)

    # scaling
    #xL = xL / np.sqrt(alpha)
    #xR = xR / np.sqrt(alpha)
    #print(xL)
    L = 2.0 * pi / (xR - xL)
    x = x/L + xL
    print(x)
    #x = np.sqrt(alpha) * x
    #x = x * np.sqrt(alpha)
    # Initial conditions

    v = f0(x)
    t = 0

    #xR = xR / np.sqrt(alpha)

    # Setting up Plot
    tplot = 0.5
    plotgap = int(round(tplot / dt))
    nplots = int(round(tmax / tplot))

    LX = len(x)
    data = np.zeros([nplots + 1, LX])
    data[0, :] = v
    tdata = np.zeros(nplots + 1)

    for i in range(1, nplots+1):

        for n in range(plotgap):  # Euler
            t = t + dt
            # Euler
            #v = euler(v, D1, D2, dt, alpha, L)


            # RK4
            #a = L**2 * alpha * P(v, D2) - L * W(v, D1)
            #b = L**2 * alpha * P(v + 0.5 * a * dt, D2) - L * W(v + 0.5 * a * dt, D1)
            #c = L**2 * alpha * P(v + 0.5 * b * dt, D2) - L * W(v + 0.5 * b * dt, D1)
            #d = L**2 * alpha * P(v + c * dt, D2) - L * W(v + c * dt, D1)
            #v = v + dt * (a + 2 * (b + c) + d) / 6.0
        u = np.zeros(len(x))
        for j in range(len(x)):

            #FL = S[np.where(S < 0)][0]
            #FR = S[np.where(S > 0)][0]
            #uj = optimize.bisect(F, FL, FR, args=(x[j], t))
            uj = optimize.root(F, 1.0, args=(x[j], t), tol=10**-10)
            u[j] = uj.x
        v = f0(u)
        data[i, :] = v
        tdata[i] = t

    return tdata, data, x


N = 512
xL = -60
xR = 60
tmax = 16.0
dt = 0.01
alpha = 0.01

tdata, data, X = Simulation(N, xL, xR, tmax, dt, alpha)


L = 2.0 * pi / (xR - xL)

h= 2.0 * np.pi / N



plt.figure(1)
plt.plot(X, data[len(tdata) - 1, :])
Z = data
#Z = np.zeros([len(tdata), len(X)])
#for j in range(0, len(tdata)):
#    ut = data[j, :]
#    Z[j, :] = fourier(X / sqrt(alpha), ut, N, X, h)

X, Y = np.meshgrid(X, tdata)
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.1)

#ax.set_zlim(0, 1.0)
ax.set_xlabel(r'$x$', fontsize=15)
ax.set_ylabel(r'$t$', fontsize=15)
ax.set_zlabel(r'$u$', fontsize=15)

plt.show()
