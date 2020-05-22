import numpy as np
from scipy.integrate import odeint
from numba import jit

from Differentiation_Matrix import fourdif

def f0(x):

    return np.exp(- 0.05 * x**2)


@jit
def F(v, t):

    Fv = L ** 2 * alpha * np.dot(D2, v) - 0.5 * L * np.dot(D1, v ** 2)


    return Fv


def Exact_Solution():

    v0 = f0(x)

    sol_exact = odeint(F, v0, tdata, rtol=1.49012e-16)

    return sol_exact


alphas = [1.0, 0.5, 0.025, 0.01, 0.005]

N = 4097
xL = -60
xR = 60
tmax = 100.0

# Grid
L = 2.0 * np.pi / (xR - xL)

x, D1 = fourdif(N, 1)
x, D2 = fourdif(N, 2)

x = x / L + xL
tdata = np.linspace(0, 100, 201)


for j in range(0, len(alphas)):
    alpha = alphas[j]
    sol_exact = Exact_Solution()
    name = 'sol_exact_' + str(alpha) + '_4097'
    np.save(name, sol_exact)
