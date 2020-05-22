from scipy.special import erfc
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

dt = 0.5 # grid size for time (s)
dx = 0.5 # grid size for space (m)
alpha = 1.0 # kinematic viscosity of oil (m2/s)
x_max = 60.0 # in m
t_max = 10.0 # total time in s
x = np.arange(-60.0,x_max+dx,dx)
t = np.arange(0, t_max + dt, dt)

def f0(x):

    return np.exp(-0.005*x**2) # velocity in m/s

def G(x, t, alpha):

    eta = 4.0 * alpha * t
    return np.exp(- x**2 / eta)

# using the analytic solution:
def diffusion_analytic(alpha):
    h = 1.0
    y = np.arange(-200, 200 + h, h)
    data = np.zeros([len(t), len(x)])
    data[0, :] = f0(x)

    for j in range(len(x)):
        xj = x[j]

        for i in range(1, len(t)):
            ti = t[i]

            eps = (1.0 / np.sqrt(4.0 * alpha * ti * np.pi))
            sum1 = 0.0
            for k in range(1, len(y) - 1):
                sum1 = sum1 + G(xj - y[k], ti, alpha) * f0(y[k])

            Ga = G(xj - y[0], ti, alpha) * f0(y[0])
            Gb = G(xj - y[len(y) - 1], ti, alpha) * f0(y[len(y) - 1])
            sum1 = sum1 + 0.5 * (Ga + Gb)
            data[i, j] = eps * sum1

    return data



# plotting:
X, Y = np.meshgrid(x, t)
Z = diffusion_analytic(alpha)
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