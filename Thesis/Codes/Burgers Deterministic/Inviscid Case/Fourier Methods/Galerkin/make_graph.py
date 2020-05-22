import numpy as np

import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import LineCollection

from ExactSol_Burgers import exact
from Fourier_Galerkin import Simulation

a = -5.0
b = 5.0
N = 2000
x = np.linspace(a, b, 200)
tf = 10
P = int(tf / 0.015)
t = np.array([0.015 * i for i in range(0, P)])


def exact_solution():

    fig = plt.figure(figsize=plt.figaspect(0.5))

    sol1 = exact(x, t, N, 0.1)
    sol2 = exact(x, t, N, 0.01)

    X, Y = np.meshgrid(x, t)
    Z = sol1
    W = sol2

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0.1)

    ax.set_zlim(0, 1.0)
    ax.set_xlabel(r'$x$', fontsize=15)
    ax.set_ylabel(r'$t$', fontsize=15)
    ax.set_zlabel(r'$u$', fontsize=15)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, W, cmap=cm.jet, linewidth=0.1)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel(r'$x$', fontsize=15)
    ax.set_ylabel(r'$t$', fontsize=15)
    ax.set_zlabel(r'$u$', fontsize=15)

    np.savetxt('Graphics/' + 'sol_0.1', Z)
    np.savetxt('Graphics/' + 'sol_0.01', W)

    #fig.savefig('Graphics/' + 'Exact_solution')


def fourier():
    # Grid
    N = 128
    xL = -1.5
    xR = 3.0
    tmax = 10.0

    tdata, data, X = Simulation(N, xL, xR, tmax)

    X, Y = np.meshgrid(X, tdata)
    Z = data

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.1)

    ax.set_zlim(0, 1.0)
    ax.set_xlabel(r'$x$', fontsize=15)
    ax.set_ylabel(r'$t$', fontsize=15)
    ax.set_zlabel(r'$u$', fontsize=15)

    '''''
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca(projection='3d')
    poly = LineCollection(data)
    poly.set_alpha(0.5)
    ax.add_collection3d(poly, zs=tdata, zdir='y')
    ax.set_xlabel('X')
    ax.set_xlim3d(xL, xR)
    ax.set_ylabel('Y')
    ax.set_ylim3d(0, tmax)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 1.0)
    ax.view_init(75, -85)
    ax.set_title('N = ' + str(N))
    '''''


def compare():
    N = 128
    xL = -5.0
    xR = 5.0
    tmax = 10.0

    sol_exact = np.loadtxt('Graphics/' + 'sol_0.1')
    plt.figure()

    tdata, data, X = Simulation(N, xL, xR, tmax)

    sol_t = sol_exact[0, :]
    plt.plot(x, sol_t)
    plt.plot(X, data[0, :], '--')
    print(tdata)


#exact_solution()

print(t)
#Chebychev()
#print(t)
#compare()

plt.show()