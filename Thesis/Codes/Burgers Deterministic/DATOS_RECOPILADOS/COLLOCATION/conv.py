import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from sympy import symbols
from scipy.integrate import trapz

def f(x, b):
    return np.exp(-abs(x)**2/b)

def g(x):

    return np.exp(-0.05*x**2)


alpha = 0.1
t=10
b = 4 * t * alpha

x = np.linspace(-60,60,100)
dx = x[1] - x[0]

h1 = fftconvolve(f(x, b),g(x),mode='same')* dx / np.sqrt(4 * np.pi * t * alpha)

plt.figure()
plt.plot(x,h1,label='fftconvolve')

plt.legend()
plt.show()