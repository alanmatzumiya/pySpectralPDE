import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def u0(x):

    return np.exp(-(2.0 * (x - 1.0))**2)


def trap(f, N, a, b):
    h = (b - a) / N
    sum = 0
    for j in range(0, N):
        xj = a + j * h
        sum = 2.0 * f(xj)
    integral = (h / 2.0) * (f(a) + sum + f(b))
    return integral


def phi0(u0, z, eps, N):
    int0 = []
    for i in range(0, len(z)):
        zi = z[i]
        h = zi / N
        sum = 0
        for j in range(1, N):
            zj = j * h
            sum = sum + 2.0 * u0(zj)
        integral = (h / 2.0) * (u0(0) + sum + u0(zi))
        I0 = np.exp(- integral / (2 * eps))
        int0.append(I0)

    return int0


def integ2(phiz, xi, zi, tj, eps):

    return ((xi - zi) / tj) * phiz * np.exp(- (xi - zi)**2 / (4.0 * eps * tj))


def integ1(phiz, xi, zi, tj, eps):

    return phiz * np.exp(- (xi - zi)**2 / (4.0 * eps * tj))


def exact(x, t, N, eps):
    M = 10
    z = np.linspace(-M, M, N)
    phi = phi0(u0, z,eps, N)
    LX = len(x)
    LT = len(t)
    TX = np.zeros([LT, LX])
    TX[0, :] = u0(x)
    h = abs(z[1] - z[0])
    for j in range(1, len(t)):
        tj = t[j]
        for i in range(0, len(x)):
            xi = x[i]
            sum1 = 0 ; sum2 = 0
            for k in range(1, len(z) - 1):
                zi = z[k]
                fun1 = integ1(phi[k], xi, zi, tj, eps)
                fun2 = integ2(phi[k], xi, zi, tj, eps)
                sum1 = sum1 + 2.0 * fun1
                sum2 = sum2 + 2.0 * fun2
            int1 = (h / 2.0) * (integ1(phi[0], xi, z[0], tj, eps) + sum1 + integ1(phi[len(z) - 1], xi, z[len(z) - 1], tj, eps))
            int2 = (h / 2.0) * (integ2(phi[0], xi, z[0], tj, eps) + sum2 + integ2(phi[len(z) - 1], xi, z[len(z) - 1], tj, eps))
            u_tx = int2 / int1
            TX[j, i] = u_tx

    return TX