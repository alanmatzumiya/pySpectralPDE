import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.integrate import quad, dblquad, romberg, trapz
from scipy.special import erf, eval_hermitenorm
from numpy import polynomial
import math

from matplotlib.ticker import LinearLocator, FormatStrFormatter


def u0(x):

    return np.exp(-0.005 * x**2).astype(np.float128)


def integrand1(xi, tj, eps, rulesX, rulesW):
    z = xi - rulesX * np.sqrt(2.0 * eps * tj)
    z = z.astype(np.float64)
    '''''
    integ1 = np.zeros(len(z))
    for k in range(len(z)):
        zk = z[k]
        integ1[k] = quad(u0, 0, zk)[0]
    '''''
    integ1 = erf(z * np.sqrt(0.005)) * (np.sqrt(np.pi) / (2.0 * np.sqrt(0.005)))
    integ1 = integ1.astype(np.float128)

    sum1=0
    for i in range(len(z)):
        C = np.exp(-integ1[i] / (2.0 * eps)).astype(np.float128)
        if math.isinf(C):
            sum1 = sum1 + 0.0

        else:
            sum1 = sum1 - np.sqrt(2.0 * eps * tj) * C * rulesW[i]
    return sum1


    #return np.exp(-(int1 / (2.0 * eps) + (xi - z)**2 / (4.0 * eps * tj)))

def integrand2(xi, tj, eps, rulesX, rulesW):

    z = xi - rulesX * np.sqrt(2.0 * eps * tj)
    z = z.astype(np.float64)
    '''''
    integ2 = np.zeros(len(z))
    for k in range(len(z)):
        zk = z[k]
        integ2[k] = quad(u0, 0, zk)[0]
    '''''
    integ2 = erf(z * np.sqrt(0.005)) * (np.sqrt(np.pi) / (2.0 * np.sqrt(0.005)))
    integ2 = integ2.astype(np.float128)

    sum2 = 0
    for i in range(len(z)):
        C = np.exp(-integ2[i] / (2.0 * eps)).astype(np.float128)

        if math.isinf(C):

            sum2 = sum2 + 0.0
        else:
            sum2 = sum2 - rulesX[i] * 2.0 * eps * tj * C * rulesW[i]
    return sum2

    #return ((xi - z) / tj) * np.exp(-(integ2 / (2.0 * eps) + (xi - z)**2 / (4.0 * eps * tj)))


def exact(eps, x, t):

    #dz = 1.0
    #M = 3 * max(x)
    #rulesX = np.arange(-M, M + dz, dz).astype(np.float128)
    Q = 250
    rule1 = np.polynomial.hermite_e.hermegauss(Q)
    #rulesW = np.polynomial.hermite_e.hermeweight(rulesX).astype(np.float128)
    rulesX = rule1[0].astype(np.float128)[::-1]
    rulesW = rule1[1].astype(np.float128)
    #phiz = np.zeros(len(z))
    #for i in range(len(z)):
    #    phiz[i] = integrand(z[i], eps)
    LX = len(x)
    if isinstance(t, float):
        t = [0.0, t]
    LT = len(t)
    TX = np.zeros([LT, LX], dtype=np.float128)
    TX[0, :] = u0(x)

    for j in range(1, len(t)):
        tj = t[j]
        for i in range(0, len(x)):
            xi = x[i]
            #vec_F1 = np.vectorize(F1)
            #vec_F2 = np.vectorize(F2)
            #int1 = F1(xi, tj, eps)
            #int2 = F2(xi, tj, eps)
            #int1 = vec_F1(xi, tj, eps)
            #int2 = vec_F2(xi, tj, eps)
            '''''
            sum1 = 0 ; sum2 = 0
            for k in range(1, len(z) - 1):
                zi = z[k]
                fun1 = integrand1(zi, xi, tj, eps)

                fun2 = ((xi - zi) / tj) * fun1
                sum1 = sum1 + fun1
                sum2 = sum2 + fun2
            Ga = integrand1(z[0], xi, tj, eps)
            Gb = integrand1(z[len(z) - 1], xi, tj, eps)
            int1 = dz * (sum1 + 0.5 * (Ga + Gb))

            Sa = ((xi - zi) / tj) * Ga
            Sb = ((xi - zi) / tj) * Gb
            int2 = dz * (sum2 + 0.5 * (Sa + Sb))
            '''''
            int1 = integrand1(xi, tj, eps, rulesX, rulesW)

            int2 = integrand2(xi, tj, eps, rulesX, rulesW)


            #if (int1 or int1) == 0.0:
                #u_tx = 0.0
            #else:
            try:
                u_tx = abs(int2 / int1)

            except RuntimeWarning and ZeroDivisionError:
                u_tx = 0.0


            TX[j, i] = u_tx / tj
            #except RuntimeWarning:
            #    pass

    return TX

'''''
a = -60; b = 60
N = 128
hx = 2.0 * np.pi / N  # step size
p = 2 * np.pi / (b - a)
x = hx * np.arange(0, N, dtype=np.float128)
x = x / p + a

dt = 0.5 # grid size for time (s)

alpha = 0.01# kinematic viscosity of oil (m2/s)

t_max = 10.0 # total time in s

t = np.arange(1, t_max + dt, dt, dtype=np.float128)

# plotting:
X, Y = np.meshgrid(x, t)
Z = sol2 = exact(alpha, x, t)

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
'''''