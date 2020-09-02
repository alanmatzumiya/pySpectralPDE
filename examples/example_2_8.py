import numpy as np
from numpy import exp
import matplotlib.pyplot as plt


def u(x):

    return np.sin(x / 2)


def IN(x, z, u, N):
    n = np.arange(-N / 2, N / 2 + 1)
    cn = []
    u_hat = []

    for i in range(len(n)):

        if abs(n[i]) == N / 2:
            ci = (1.0 / (2 * N)) * sum(u(x) * exp(-1j * n[i] * x))
            cn.append(ci)

        else:
            ci = (1.0 / N) * sum(u(x) * exp(-1j * n[i] * x))
            cn.append(ci)

    for k in range(len(z)):
        ui = sum(cn * np.exp(1j * n * z[k]))
        u_hat.append(ui.real)

    return u_hat

M=100
T = 2 * np.pi
z=np.linspace(0, 2.0 * np.pi, M)

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].plot(z / T, u(z))

legend = [r'$u(x)$']
for N in 2**np.arange(2, 5):
    x = []
    for j in range(N):
        xj = 2.0 * np.pi * j / N
        x.append(xj)
    x = np.array(x)
    u_approx = IN(x, z, u, N)
    ax[0].plot(z / T, u_approx, '--')
    legend = legend + ['N' + ' = ' + str(N)]

ax[0].set_xlim(0, 1.0)
ax[0].set_ylim(0, 1.1)
ax[0].legend(legend)
ax[0].set_xlabel(r'$x / 2 \pi$', fontsize=13)
ax[0].set_ylabel(r'$\mathcal{I}_N u$', fontsize=13)
ax[0].set_title('(a)', loc='left')

legend = []
for N in 2**np.arange(2, 7):
    x = []
    z = []
    for j in range(N):
        xj = 2.0 * np.pi * j / N
        x.append(xj)
    M = 200 * N
    for i in range(M):
        zi = 2.0 * np.pi * i / M
        z.append(zi)
    z = np.array(z)
    x = np.array(x)

    u_approx = IN(x, z, u, N)
    Error = []
    cota=min(abs(u(z) - u_approx))
    for k in range(len(z)):
        ei = abs(u(z[k]) - u_approx[k])
        if ei < cota:
            ei = 0
            Error.append(ei)
        Error.append(np.log(ei))
    Error = np.array(Error)
    ax[1].plot(z / T, Error, '--')
    legend = legend + ['N' + ' = ' + str(N)]

ax[1].set_xlim(0, 1)
ax[1].set_ylim(-33, 0)
ax[1].legend(legend)
ax[1].set_xlabel(r'$x / 2 \pi$', fontsize=13)
ax[1].set_ylabel(r'$\ln |u - \mathcal{I}_N u|$', fontsize=13)
ax[1].set_title('(b)', loc='left')

fig.savefig('example28')

plt.show()



