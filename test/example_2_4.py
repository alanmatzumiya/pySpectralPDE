import numpy as np
import matplotlib.pyplot as plt


def u(x):

    u = np.sin(x / 2.0)

    return u

def u_hat(N, x):

    u_n = (2 / np.pi) / (1.0 - 4.0 * (np.arange(-N/2, N/2 + 1))**2)
    U = []
    for xj in x:
        U_j = sum(u_n * np.exp(1j * np.arange(-N/2, N/2 + 1) * xj)).real
        U.append(U_j)
    return U


x = np.linspace(0, 2 *np.pi, 1000)
z = np.linspace(0, 2 *np.pi, 1000)
T = 2 * np.pi

fig, ax = plt.subplots(1, 2, figsize=(10,5))

#figure1 = plt.figure(1)
ax[0].plot(x / T, u(x))

legend = [r'$u(x)$']
for N in 2**np.arange(2, 5):

    U = u_hat(N, z)

    ax[0].plot(z / T, U, '--')
    legend = legend + ['N' + ' = ' + str(N)]

ax[0].set_xlim(0, 1.0)
ax[0].set_ylim(0, 1.1)
ax[0].legend(legend)
ax[0].set_xlabel(r'$x / 2 \pi$', fontsize=13)
ax[0].set_ylabel(r'$\mathcal{P}_N u$', fontsize=13)
ax[0].set_title('(a)', loc='left')

#figure2 = plt.figure(2)

legend = []
for M in 2**np.arange(2, 7):
    ax[1].plot(z / T, np.log(abs(u(z) - u_hat(M, z))), '--')
    legend = legend + ['N' + ' = ' + str(M)]

ax[1].set_xlim(0, 1)
ax[1].legend(legend)
ax[1].set_xlabel(r'$x / 2 \pi$', fontsize=13)
ax[1].set_ylabel(r'$\ln |u - \mathcal{P}_N u|$', fontsize=13)
ax[1].set_title('(b)', loc='left')

#figure1.savefig('Approximation')
#figure2.savefig('Error')

fig.savefig('example24')

plt.show()

