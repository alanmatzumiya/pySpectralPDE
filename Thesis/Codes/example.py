from JNM import Js
from numpy import sin, pi, sqrt, exp, cos
import numpy as np
from scipy.integrate import dblquad
from scipy import special
import matplotlib.pyplot as plt


def measure(x):

    mu = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x**2)
    return mu

N = 5
M = 11
J = Js(N)


def integ(J):
    Pn = np.zeros([M, M])
    Qn = np.zeros([M, M])
    for j in range(0, M):
        for i in range(0, M):
            if (J[i, j] > 0):
                Jij = int(J[i, j])
                P = lambda x, y: (abs(special.eval_hermitenorm(Jij, x) - special.eval_hermitenorm(Jij, y))** 2) * measure(x) * measure(y)
                Q = lambda x, y: (abs(special.eval_hermitenorm(Jij, y))** 2) * measure(x) * measure(y)
                integn1 = dblquad(P, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
                integn2 = dblquad(Q, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
                Pn[i, j] = integn1[0]
                Qn[i, j] = integn2[0]
    return Pn, Qn

print(integ(J))

  	




