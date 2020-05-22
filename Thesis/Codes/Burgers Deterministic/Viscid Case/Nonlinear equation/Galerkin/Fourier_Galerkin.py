import numpy as np
from numpy import pi, round
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm

# initial condition function
def f0(x):

    return np.exp(- 0.05 * x ** 2)


# nonlinear term scheme 1
def W(v, v_hat, ik, p):

    u_hat = p * ik * v_hat
    uv = np.real(np.fft.ifft(u_hat)) * v
    uv_hat = np.fft.fft(uv)
    return uv_hat


# nonlinear term scheme 2
def R(v, ik, p):

    v2_hat = np.fft.fft(v ** 2)
    w = 0.5 * p * ik * v2_hat
    return w


# time numerical integrators
def forward_euler(alpha, v, v_hat, dt, p, ik, k2):
    v_hat = v_hat + dt * (p ** 2 * alpha * k2 * v_hat - R(v, ik, p))

    return v_hat


def backward_euler(alpha, v_old, v_hat_old, dt, p, ik, k2):

    v_hat_test = forward_euler(alpha, v_old, v_hat_old, dt, p, ik, k2)
    L = p ** 2 * alpha * k2 * v_hat_old * dt
    error = 1.0
    tol = 10.0 ** -10
    while error > tol:
        v_test = np.real(np.fft.ifft(v_hat_test))
        v_hat_new = v_hat_old + L - dt * R(v_test, ik, p)
        error = max(abs(v_hat_new - v_hat_test))

        v_hat_test = v_hat_new

    return v_hat_test


# numerical method
def Simulation(alpha, N, xL, xR, tmax, dt):
    # space grid
    h = 2.0 * np.pi / N
    p = 2 * pi / (xR - xL)
    x = np.array([h*i for i in range(1, N+1)])
    x = x/p + xL

    # initial condition
    t = 0
    v = f0(x)

    # wave coefficients
    k = np.zeros(N)
    k[0: int(N/2)] = np.arange(0, int(N/2))
    k[int(N/2):] = np.arange(-int(N/2), 0)
    ik = 1j * k
    k2 = ik**2

    # setting up plot
    tdata = np.linspace(0, tmax, int(round(tmax / dt)) + 1)
    LX = len(x)
    tplot = 0.5
    plotgap = int(round(tplot / dt))
    nplots = int(round(tmax / tplot))

    # data storage
    data = np.zeros([nplots + 1, LX])
    tdata = np.zeros(nplots + 1)
    data[0, :] = v

    # numerical solutions
    for i in range(1, nplots + 1):
        v_hat = np.fft.fft(v)
        for n in range(plotgap):
            t = t + dt
            #v_hat = forward_euler(alpha, v, v_hat, dt, p, ik, k2)
            v_hat = backward_euler(alpha, v, v_hat, dt, p, ik, k2)
            v = np.real(np.fft.ifft(v_hat))
        data[i, :] = v
        tdata[i] = t
        if np.isnan(v).any():
            break


    return tdata, data, x


# plot graphics
def plot(X, tdata, data):

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


# paramereters
N = 24
xL = -60.0
xR = 60.0
tmax = 100.0
dt = 0.01
alpha = 0.01

ti = time()  # initial time execution
tdata, data, X = Simulation(alpha, N, xL, xR, tmax, dt)
tf = time() - ti; print(tf)  # final time execution

plot(X, tdata, data)
