from numpy import sin, pi, sqrt, exp, cos
import numpy as np
import itertools
from math import factorial
import scipy.integrate as integrate
from scipy.integrate import dblquad
from scipy import special

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, show
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator


from numba import jit



import pylab as pl


def u0(x):
    return sin(pi * x)

def taylor(x):
    aprox = 0
    for j in range(1, n + 1):
        aprox = aprox + ((-1)**(j - 1) * (pi * x) ** (2 * j - 1)) / factorial(2*j - 1)
    return aprox

def J_NM(N, M, N1):

    J = np.zeros([M, M])
    L1 = 5

    for k in range(1, N1):
        h = list(itertools.combinations(range(3, N + 1), k))
        L = len(h)
        L2 = L1 + L - 1
        J[0, L1-1:L2] = 1
        J[1, L1-1:L2] = 2
        for j in range(0, L):
            for i in range(0, len(h[j])):
                J[2  + i , L1 + j - 1] = h[j][i]
        L1 = L1 + L

    J[0, 0] = 2
    J[0, 1] = 1
    J[1, 1] = 2

    J[0, 2] = 1
    J[1, 2] = 2
    J[2, 2] = 1

    J[0, 3] = 1
    J[1, 3] = 2
    J[2, 3] = 2
    return J

def Js(N):
    M = 0
    if (N == 4):
        M = 7
        N1 = 3
        J1 = J_NM(N, M, N1)

    if (N == 5):
        M = 11
        N1 = 4
        J1 = J_NM(N, M, N1)

    if (N == 6):
        M = 18
        N1 = 4
        J2 = J_NM(N, M, N1)
        SeqJ = np.array([1, 3] + range(5,7) + [8] +  range(11,14) + [15, 17]) - 1  #### This is with the numbers (N=6, M=18, N1=4),
        M = 10
        J1 = J2[0:M, SeqJ]

    if (N == 7):
        M = 29
        N1 = 4
        J2 = J_NM(N, M, N1)
        SeqJ = np.array(range(1, 10) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29]) - 1  #### This is with the numbers (N=7, M=29, N1=4),
        M = 20
        print(len(SeqJ))
        J1 = J2[0:M, SeqJ]

    if (N == 8):
        M = 60
        N1 = 5
        J2 = J_NM(N, M, N1)
        SeqJ = np.array(range(1,10) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29, 31, 34, 35, 38, 39, 44, 48, 51, 55, 59]) - 1  #### This is with the numbers (N=7, M=29, N1=4),
        M = 30
        J1 = J2[0:M, SeqJ]

    if (N == 9):
        M = 150
        N1 = 5
        J2 = J_NM(N, M, N1)
        SeqJ = np.array(range(1, 10) + [11, 12, 13, 16, 18, 19, 23, 24, 25, 28, 29, 31, 34, 35, 38, 39, 44, 48, 51, 55, 59, 63, 67, 70, 75, 81, 85, 89, 92, 95, 99]) - 1  #### This is with the numbers (N=7, M=29, N1=4),
        M = 40
        J1 = J2[0:M, SeqJ]

    return J1


def evalXSin(nu, k, u0):
    f = lambda x: sqrt(2.0 * nu) * k * pi * sqrt(2.0 / pi) * (u0(x)) * (sin(k * pi * x))  # *exp(-((sin(pi*x))^2)/2)/sqrt(2*pi)}  ##

    integr = integrate.quad(f, 0, 1)  ### 1

    int = integr[0]

    return (int)

def integr1(J, M, rulesX, rulesW, LRules, nu, u0):
    intg = 0
    for k in range(0,LRules):
        prod = 1
        for i in range(0,M):
            if (J[i,0] > 0):
                x1 = evalXSin(nu, i + 1, u0)
                prod = prod * special.eval_hermitenorm(int(J[i,0]), rulesX[k])  ##(1/sqrt(factorial(J1N[i])))*
        intg = intg + prod * rulesW[k]

    return (intg)

def integr2(Jm, Jn,M,rulesX,rulesW,LRules,nu, r, i):
    intg = np.zeros(M)
    for k in range(0, M):  ### this is the k from the notation. represent the sum in k
        intg1 = 0

        if (Jn[k] > 0):
            sum2 = 0
            Jn1 = Js(N)[:,r]
            #Jn1[k] = 0
            Jm1 = Js(N)[:,i]
            #Jm1[k] = 0

            prod = 1.0
            for o in range(0, M):
                sumparc = 0

                for l in range(0, LRules):
                    x1 = rulesX[l]
                    factor1 = (1.0 / sqrt(factorial(int(Jn1[o])))) * (1.0 / sqrt(factorial(int(Jm1[o]))))
                    prod1 = special.eval_hermitenorm(int(Jn1[o]), x1) * special.eval_hermitenorm(int(Jm1[o]), x1) * factor1
                    sumparc = sumparc + prod1 * rulesW[l] / (sqrt(2.0 * nu) * pi * (o + 1))

                prod = prod * sumparc

            for l in range(0, LRules):
                x2 = rulesX[l]
                factor2 =(1.0 / sqrt(factorial(int(Jn[k]) - 1))) * (1.0 / sqrt(factorial(int(Jm[k]))))
                sum2 = sum2 + special.eval_hermitenorm(int(Jn[k]) - 1, x2) * special.eval_hermitenorm(int(Jm[k]), x2) * rulesW[l] * factor2

            intg1 = sum2 * prod  ##(1/sqrt(factorial(J[k])))*
        intg[k] = intg1  ###/(2*pi)

    return (intg)


def integr3k(M):

    int3 = np.zeros(M)
    for k in range(1, M + 1):

        f = lambda x: sqrt(2.0) * sin(k * pi * x) * x
        integr = integrate.quad(f, 0, 1)  ### 1

        sum = integr[0]
        int3[k-1] = sum
    return int3


def integr4(M):
    int4 = np.zeros(M)
    int3 = integr3k(M)
    for j in range(0, M):
        intM = np.zeros([M, M])
        for k in range(0, M):
            for l in range(0, M):
                f = lambda x: (2.0) * ((l + 1) * pi * sin((k + 1) * pi * x) * cos((l + 1) * pi * x) + (k + 1) * pi * sin((l + 1) * pi * x) * cos((k + 1) * pi * x)) * sin((j + 1) * pi * x)
                integr = integrate.quad(f, 0, 1)  ### 1
                intM[l, k] = integr[0] * int3[l] * int3[k]
        int4[j] = sum(intM.sum(1))


    return int4


def Cnm(J, N, M, rulesX, rulesW, LRules, nu):
    Cnm = np.zeros([M, M])
    I4 = integr4(M)
    for k in range(1, M):
        for i in range(1, M):
            I2 = integr2(J[:,k],J[:,i],M,rulesX,rulesW,LRules,nu, k, i)
            sum1 = 0
            for j in range(0, M):
                sum1 = sum1 + (j + 1) * sqrt(J[j, i]) * I4[j] * I2[j]
            Cnm[k, i] = sum1
    Cnm[0, 0] = pi * (-3.0) * I4[0]
    return (Cnm)


def ALambda1(Cs1, J, M, nu):
    Lamb1 = np.zeros([M, M])
    for i in range(0, M):
        sum1 = 0
        for j in range(0, M):
            sum1 = sum1 + J[j, i] * ((j + 1)**2) * pi**2

        Lamb1[i, i] = sum1 * nu
    ALambda1 = Cs1 - Lamb1

    return (ALambda1)

def u02(J,N,M,rulesX,rulesW,LRules,xSpace,EigVecRe,nu,u0):
  Lx = len(xSpace)
  ci = np.zeros([M,Lx])

  for z in range(0, Lx):
      for i in range(0, M):
          sum2 = 0
          for k in range(0, M):
              sum1 = 0
              if(J[k,i]>0):
                  for y in range(0, LRules):
                      sum1 = sum1 + special.eval_hermitenorm(int(J[k,i]), rulesX[y]) * rulesX[y] * rulesW[y]
              sum2 = sum2 + sum1 * sin(pi * (k + 1) * xSpace[z]) * sqrt(2.0)/((sqrt(2.0 * nu) * pi * (k + 1)))
          ci[i,z] = sum2
  return(ci)

def I0(J, M, EigVecRe, EigVecIm, u0, rulesX, rulesW, LRules):
    uj = np.zeros(M)
    for j in range(0, M):
        intj = 0
    for k in range(0, LRules):
        prod = 1.0
        for i in range(0, M):
            prod = prod * special.eval_hermitenorm(int(J[i, j]), rulesX[k]) * (1 / sqrt(factorial(J[i, j])))
            intj = intj + prod * u0(rulesX[k]) * rulesW[k]

        uj[j] = intj

    L = np.where(EigVecIm.sum(0) != 0)
    l = len(L[0])
    Matr = EigVecRe

    if (l > 0):
        for i in range(0, (l / 2)):
            Matr[:, L[0][2 * i]] = EigVecIm[:, L[0][2 * i]]

    I0 = np.linalg.solve(Matr, uj)
    return (I0)

# Function to simulate X.
def SimulaX(J, M, xSpace, nu, u0):

    Px = len(xSpace)
    H = np.zeros([M, Px])

    for k in range(0, Px):
        for j in range(1, M):
            prod = 1.0
            for i in range(0, M):
                if (J[i, j] > 0):
                    x1 = evalXSin(nu, i + 1, u0)
                    prod =  prod * special.eval_hermitenorm(int(J[i, j]), x1) * (1.0 / sqrt(factorial(J[i, j])))

            H[j, k] = prod
        x2 = xSpace[k]
        H[0, k] = special.eval_hermitenorm(2, x2) - special.eval_hermitenorm(1, x2) + 1

    return (H)


def SimulaT(tim, J, M, EigValRe, EigValIm, EigVecRe, EigVecIm, cons):
    Pt = len(tim)
    T = np.zeros([Pt,M])
    for k in range(0, Pt):  ### for each point t
        prod = np.zeros(M)

        for j in range(0, M):  ### for each multi-index n_j\in J^{N,M}
            sum = 0
            for i in range(0, M):  ### for each member inside the multi-index n_j=(\alpha_i,,,,)
                if (EigValIm[i] != 0):  ## complex eigenvalues
                    if (i == 1): ## first member
                        sum = sum + cons[i] * exp(EigValRe[i] * tim[k] / pi**2) * (EigVecRe[j, i] * cos(EigValIm[i] * tim[k] / pi**2) - EigVecIm[j, i] * sin(EigValIm[i] * tim[k] / pi**2))
                if (i > 1):  ##  ( first member )^C
                    if (EigValIm[i] != -EigValIm[i - 1]):
                        sum = sum + cons[i] * exp(EigValRe[i] * tim[k] / pi**2) * (EigVecRe[j, i] * cos(EigValIm[i] * tim[k] / pi**2) - EigVecIm[j, i] * sin(EigValIm[i] * tim[k] / pi**2))

                if (EigValIm[i] == -EigValIm[i - 1]):
                    sum = sum + cons[i] * exp(EigValRe[i] * tim[k] / pi**2) * (EigVecRe[j, i - 1] * sin(EigValIm[i - 1] * tim[k] / pi**2) + EigVecIm[j, i - 1] * cos(EigValIm[i - 1] * tim[k] / pi**2))

                if (EigValIm[i] == 0):
                    sum = sum + cons[i] * EigVecRe[j, i] * exp(EigValRe[i] * tim[k] / pi**2)  # *nu/pi
            prod[j] = sum
        T[k,:] = prod

    return (T)

def SimulaTX(tim, J, M, N, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, nu, H1):
    LX = len(xSpace)
    LT = len(tim)
    TX1 = np.zeros([LT, LX])
    TX = np.zeros([LT, LX])

    for x in range(0, LX):
        U1 = np.dot(np.linalg.inv(EigVecRe), U_1[:, x])  # rowMeans(U
        Time1 = SimulaT(tim, J, M, EigValRe,EigValIm,EigVecRe,EigVecIm,U1)
        TX1 = np.dot(Time1, H1)
        TX[:, x] = TX1[:, x]

    return TX


def Un(tim, J, M, N, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, nu, x):

    U1 = np.dot(np.linalg.inv(EigVecRe), U_1[:, x])  # rowMeans(U
    Time1 = SimulaT(tim, J, M, EigValRe, EigValIm, EigVecRe, EigVecIm, U1)

    return Time1



### The MAIN part of the first section of the code
### The space X and the time T is created here
N = 5

xSpace = np.linspace(0,1,129)
tim = np.linspace(0,3,101)
J = Js(N)
M = len(J[:,1])

nu = 0.01
Q = 200
rule1 = np.polynomial.hermite_e.hermegauss(Q)
rulesX = rule1[0][::-1]
rulesW = rule1[1]

# Rule <- mak_rules(rulesX1,rulesW1)
# rulesX<-Rule[,1]
# rulesW<-Rule[,2]
LRules = len(rulesX)

### The value for the maximum polynomial order : N
### Tthe value of M, This need to be modified according to the number N :
### i.e. is the sum of N!/k!(N-k)! from k=2,..,N-1,

Cs = Cnm(J,N,M,rulesX,rulesW,LRules,nu) * (sqrt(2.0 * nu) * pi)

A = ALambda1(Cs,J,M,nu)
Eig1 = np.linalg.eig(A)


EigValRe = Eig1[0].real
EigValIm = Eig1[0].imag


EigVecRe = Eig1[1].real
EigVecIm = Eig1[1].imag

H1 = SimulaX(J,M,xSpace,nu, u0)
U_1 = u02(J,N,M,rulesX,rulesW,LRules,xSpace,EigVecRe,nu,u0)

#H2 = SimulaX(J,M,xSpace,nu, taylor)
#U_2 = u02(J,N,M,rulesX,rulesW,LRules,xSpace,EigVecRe,nu,taylor)

simulation1 = abs(SimulaTX(tim,J,M,N,EigValRe,EigValIm,EigVecRe,EigVecIm,U_1,nu,H1))



n = 3

def measure(x):

    mu = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * x**2)
    return mu


def P(x, y):
    Pnx = special.eval_hermitenorm(int(J[i, j]), x)
    Pny = special.eval_hermitenorm(int(J[i, j]), y)
    h = ((abs(Pnx - Pny))**2) * measure(x) * measure(y)
    return h

def err(J, M, xSpace, nu, u0, taylor, cons):


    return (sum)
'''''
for z in range(1, 2):

    normas = []
    times = []

    UN = Un(tim, J, M, N, EigValRe, EigValIm, EigVecRe, EigVecIm, U_1, nu, z)
    VN = Un(tim, J, M, N, EigValRe, EigValIm, EigVecRe, EigVecIm, U_2, nu, z)

    for k in range(0,len(tim)):
        Ut = UN[k,:]

        for j in range(0, M):
            prod = 1.0
            sum = 0
            for i in range(0, M):
                if (J[i, j] > 0):

                    integn = dblquad(P, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
                prod = prod * integn[0]
            sum = sum + (abs(Ut[j])**2) * prod
        norma1 = sum
        normas.append(norma1)
        times.append(tim[k])

    plt.figure(z)
    plt.plot(times, normas)
'''''

#TSim = SimulaT(tim,J,M,EigValRe,EigValIm,EigVecRe,EigVecIm,U_1)

#print(TSim)

# Plot
'''''
fig = figure(figsize=(12,8))
ax = fig.gca(projection='3d')
poly = LineCollection(TSim)
poly.set_alpha(0.5)
ax.add_collection3d(poly, zs=tim, zdir='y')
ax.set_xlabel('X')
#ax.set_xlim3d(0, 1.0)
ax.set_ylabel('Y')
#ax.set_ylim3d(0, len(tim))
ax.set_zlabel('Z')
#ax.set_zlim3d(-0.1, 1.2)
ax.view_init(75,-85)
ax.set_title('N = ' + str(N))[:,p]
'''''

fig1 = pl.figure(1)
ax1 = Axes3D(fig1)
X, Y = np.meshgrid(xSpace, tim)
Z = simulation1

ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')


Lx = len(xSpace)
Lt = len(tim)
Intu0 = np.zeros(Lt)

for k in range(0, Lt):
  sumP = simulation1[k,1]+simulation1[k,Lx - 1]
  for i in range(1,(Lx-1)):
    sumP = sumP+2*simulation1[k,i]
    Intu0[k]=sumP/(2.0*(Lx-1))

plt.figure(2)
plot(tim,Intu0)



#E = []

'''''
for z in 2**np.arange(1, 5):
    n = z
    Hz = SimulaX(J, M, xSpace, nu, taylor)
    U_z = u02(J, N, M, rulesX, rulesW, LRules, xSpace, EigVecRe, nu, taylor)

    simulationz = abs(SimulaTX(tim, J, M, N, EigValRe, EigValIm, EigVecRe, EigVecIm, U_z, nu, Hz))

    Intv0 = np.zeros(Lt)
    for k in range(0, Lt):
      sumP = simulationz[k,1]+simulationz[k,Lx - 1]
      for i in range(1,(Lx-1)):
        sumP = sumP+2*simulationz[k,i]
        Intv0[k]=sumP/(2.0*(Lx-1))
    err = max(abs(Intv0 -Intu0))
    E.append(err)
    plot(tim, Intv0)
'''''''''
#plt.figure(2)
#plt.semilogy(2**np.arange(1,5), E)



plt.show()
