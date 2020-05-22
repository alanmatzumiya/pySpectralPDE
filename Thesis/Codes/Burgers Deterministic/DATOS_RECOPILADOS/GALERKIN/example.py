import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quadrature, trapz


def f(x):

    return np.sin(x)


def f0(x):

    return np.exp(- 0.05 * x **2)

def f0_prime(x):

    return -np.exp(- 0.05 * x **2) * 0.1 * x


def f_prime(x):

    return np.cos(x)

def f_doubleprime(x):

    return -np.sin(x)

N = 1024; M = 1024
xL =-60; xR = 60

T = 2.0 * np.pi
hx = T / N
hz = T / M

p = 2 * np.pi / (xR - xL)
x = hx * np.arange(0, N)
z= hz * np.arange(0, M)
x = x / p + xL
z = z /p + xL

y = f0(x)
#y[N] = 0

y_hat = np.fft.fft(y)

k = np.zeros(N)
k[0: int(N/2)] = np.arange(0, int(N/2))
k[int(N/2):] = np.arange(-int(N/2), 0, 1)
print(k)

jk = p * k * 1j
k2 = p**2 * jk ** 2

def fourier(ks):
    F = np.zeros(len(z))
    for j in range(0, len(z)):

        F[j] = sum(ks * y_hat * np.exp(1j * k * hz * j)).real / N
    return F

F_0 = fourier(np.ones(N))
f_x = fourier(jk)
f_xx = fourier(k2)

G_0hat = np.fft.fft(F_0)
G_0 = np.fft.ifft(G_0hat)

print(max(abs(G_0 - F_0)))

plt.figure(1)
plt.plot(z, F_0)
plt.plot(z, f0(z), '--')
print(max(abs(F_0 - f0(z))))

plt.figure(2)
plt.plot(z, f_x)
plt.plot(z, f0_prime(z), '--')
print(max(abs(f_x - f0_prime(z))))


plt.show()