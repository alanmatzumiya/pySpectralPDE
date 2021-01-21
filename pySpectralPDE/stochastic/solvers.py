from numpy import sin, pi, sqrt, exp, cos
import numpy as np
from math import factorial
import scipy.integrate as integrate
from scipy import special


class integrator:

    def __init__(self, u0, JNM, **params):
        self.u0 = u0
        self.nu = params['nu']
        self.x = params['x']
        self.t = params['t']
        self.N = params['N']

        # Hermite polynomials evaluation
        rule1 = np.polynomial.hermite_e.hermegauss(deg=200)
        self.rulesX = rule1[0][::-1]
        self.rulesW = rule1[1]
        self.LRules = len(self.rulesX)
        self.J = JNM().Js(self.N)
        self.M = len(self.J)

        self.EigValRe, self.EigValIm, self.EigVecRe, self.EigVecIm, self.U_1 = self.EigeF()
        self.H1 = self.SimulaX()

    def integr1(self):
        """
        Function to calculate first type of integrals.

        Parameters
        ----------
        J : array; shape([M, M])
            Array containing the grades of the polynomials
        M : int; max order polynomials
        rulesX : array; polynomials of Hermite evaluated
        rulesW : array; weights of polynomials of Hermite evaluated
        LRules : array; length of rulesX array
        nu : float; Diffusion coefficient
        u0 : callable(x); initial condition function

        Returns
        -------
        intg : float, value of first type of integral

        """

        intg = 0
        for k in range(0, self.LRules):
            prod = 1
            for i in range(0, self.M):
                if self.J[i, 0] > 0:
                    prod = prod * special.eval_hermitenorm(int(self.J[i, 0]), self.rulesX[k])
            intg = intg + prod * self.rulesW[k]

        return intg

    def integr2(self, Jm, Jn, r, i):
        """
        Function to calculate second type of integrals.

        Parameters
        ----------
        Jm : int; column m of matrix J
        Jn : int; column m of matrix J
        r : int; column index of matrix J
        i : int; column index of matrix J

        Returns
        -------
        intg : array, size M
            Array containing the values of second type of integral

        """

        def f(Jm, Jn, r, i, J, M, LRules, rulesX, rulesW, nu, k):
            sum2 = 0
            Jn1 = J[:, r]

            Jm1 = J[:, i]

            prod = 1.0
            for o in range(0, M):
                sumparc = 0

                for l in range(0, LRules):
                    x1 = rulesX[l]
                    factor1 = (1.0 / sqrt(factorial(int(Jn1[o])))) * (1.0 / sqrt(factorial(int(Jm1[o]))))
                    prod1 = special.eval_hermitenorm(int(Jn1[o]), x1) * special.eval_hermitenorm(int(Jm1[o]),
                                                                                                 x1) * factor1
                    sumparc = sumparc + prod1 * rulesW[l] / (sqrt(2.0 * nu) * pi * (o + 1))

                prod = prod * sumparc

            for l in range(0, LRules):
                x2 = rulesX[l]
                factor2 = (1.0 / sqrt(factorial(int(Jn[k]) - 1))) * (1.0 / sqrt(factorial(int(Jm[k]))))
                sum2 = sum2 + special.eval_hermitenorm(int(Jn[k]) - 1,
                                                       x2) * special.eval_hermitenorm(int(Jm[k]),
                                                                                      x2) * rulesW[l] * factor2
            return sum2 * prod

        def make(f):

            def integ(Jm, Jn, r, i, J, M, LRules, rulesX, rulesW, nu):
                intg = np.zeros(M)
                for k in range(0, M):
                    if (Jn[k] > 0):
                        intg[k] = f(Jm, Jn, r, i, J, M, LRules, rulesX, rulesW, nu, k)
                    else:
                        intg[k] = 0.0
                return intg
            return integ

        return make(f)(Jm, Jn, r, i, self.J, self.M, self.LRules, self.rulesX, self.rulesW, self.nu)



    def integr3k(self):
        """
        Function to calculate third type of integrals.

        Parameters
        ----------
        M : int; size of individual timestep

        Returns
        -------
        int3 : array, size M
            Array containing the values of third type of integral

        """

        def f(x, i):
            return sqrt(2.0) * sin(i * pi * x) * x

        int3 = np.zeros(self.M)
        def make(f):

            def integ(M):
                for k in range(1, M + 1):
                    int3[k - 1] = integrate.quad(f, 0, 1, args=k)[0]
                return int3
            return integ
        return make(f)(self.M)

    def integr4(self):
        """
        Function to simulate T-X.

        Parameters
        ----------
        M : int; size of individual timestep

        Returns
        -------
        int4 : array, size M
            Array containing the values of fourth type of integral

        """


        def h(x, j, k, l):
            return 2.0 * ((l + 1) * pi * sin((k + 1) * pi * x) * cos((l + 1) * pi * x)
                          + (k + 1) * pi * sin((l + 1) * pi * x) * cos((k + 1) * pi * x)) * sin((j + 1) * pi * x)
        def w(x, j, k, l):
            return 2.0 * (sin((k + 1) * x) * sin((l + 1) * pi * x)) * sin((j + 1) * pi * x)

        int3 = self.integr3k()
        

        int4 = np.zeros(self.M)
        for j in range(0, self.M):
            intM = np.zeros([self.M, self.M])
            for k in range(0, self.M):
                for l in range(0, self.M):
                    integr = integrate.quad(h, 0, 1, args=(j, k, l))
                    intM[l, k] = integr[0] * int3[l] * int3[k]
            int4[j] = sum(intM.sum(1))
        return int4
        #else:
        #    return int3 - make(w)(self.M)

    def u02(self):
        """
        Function to calculate the constants of the system of ordinary differential equations given the initial condition

        Parameters
        ----------
        J : array; shape(M, M)
            Array containing the grades of the polynomials
        N : int; order max of the polynomials
        M : int; number of ODEs
        rulesX : array; polynomials of Hermite evaluated
        rulesW : array; weights of polynomials of Hermite evaluated
        LRules : array; length of rulesX array
        xSpace : array; discretized real space
        nu : array; Diffusion coefficient
        u0 : callable(x); initial condition function

        Returns
        -------
        ci: array, shape(M, len(xSpace))
            Array containing the constants of the system of ordinary differential equations

        """
        Lx = len(self.x)
        ci = np.zeros([self.M, Lx])

        for z in range(0, Lx):
            for i in range(0, self.M):
                sum2 = 0
                for k in range(0, self.M):
                    sum1 = 0
                    if self.J[k, i] > 0:
                        for y in range(0, self.LRules):
                            sum1 = sum1 + special.eval_hermitenorm(int(self.J[k, i]), self.rulesX[y]) * self.rulesX[y] * self.rulesW[y]
                    sum2 = sum2 + sum1 * self.u0(self.x[z]) * sqrt(2.0) / (sqrt(2.0 * self.nu) * pi * (k + 1))
                ci[i, z] = sum2
        return ci


    def Cnm(self):
        """
        Function to computes Matrix

        Parameters
        ----------
        J : array; shape(M, M)
        Array containing the grades of the polynomials
        M : int; max order polynomials
        rulesX : array; polynomials of Hermite evaluated
        rulesW : array; weigths of polynomials of Hermite evaluated
        LRules : array; length of rulesX array
        nu : array; Diffusion coefficient

        Returns
        -------
        Cnm: array, shape(len(t), len(y0))
        Array containing the value of y for each desired time in t,

        """
        Cnm = np.zeros([self.M, self.M])
        I4 = self.integr4()

        for k in range(1, self.M):
            for i in range(1, self.M):
                I2 = self.integr2(self.J[:, k], self.J[:, i], k, i)
                sum1 = 0
                for j in range(0, self.M):
                    sum1 = sum1 + (j + 1) * sqrt(self.J[j, i]) * I4[j] * I2[j]
                Cnm[k, i] = sum1
        Cnm[0, 0] = pi * (-3.0) * I4[0]
        return Cnm

    def EigeF(self):

        """
        Function to computes eigenvalues and eigenvectors of the matrix A

        Parameters
        ----------
        J : array; shape(M, M)
            Array containing the grades of the polynomials
        N : int; order max of the polynomials
        M : int; number of ODEs
        rulesX : array; polynomials of Hermite evaluated
        rulesW : array; weights of polynomials of Hermite evaluated
        LRules : array; length of rulesX array
        xSpace : array; discretized real space
        nu : array; Diffusion coefficient
        u0 : callable(x); initial condition function

        Returns
        -------
        EigValRe: array, size M
            Array containing the real eigenvalues of matrix A
        EigValIm: array, size M
            Array containing the imaginary eigenvalues of matrix A
        EigVecRe: array, shape(M, M)
            Array containing the real eigenvectors of matrix A
        EigVecIm: array, shape(M, M)
            Array containing the imaginary eigenvectors of matrix A
        U_1: array, shape(M, len(xSpace))
            Array containing the ordinary differential equations system constants

        """
        Lamb1 = np.zeros([self.M, self.M])
        for i in range(0, self.M):
            sum1 = 0
            for j in range(0, self.M):
                sum1 = sum1 + self.J[j, i] * ((j + 1) ** 2) * pi ** 2

            Lamb1[i, i] = sum1 * self.nu

        ALambda1 = self.Cnm() - Lamb1
        Eig1 = np.linalg.eig(ALambda1)

        return Eig1[0].real, Eig1[0].imag, Eig1[1].real, Eig1[1].imag, self.u02()

    def SimulaX(self):
        """
        Function to simulate X.

        Parameters
        ----------
        J : array; shape([M, M])
            Array containing the grades of the polynomials
        M : int; number of ODEs
        xSpace : array; discretized real space
        nu : array; Diffusion coefficient
        u0 : callable(x); initial condition function

        Returns
        -------
        H : array, shape(M, len(xSpace))
            Array containing hermite polynomials evaluated to simulate real space

        """

        def evalXSin(x, k):

            return sqrt(2.0 * self.nu) * k * pi * sqrt(2.0 / pi) * (self.u0(x)) * (sin(k * pi * x))
        
        Px = len(self.x)
        H = np.zeros([self.M, Px])
        for k in range(0, Px):
            for j in range(1, self.M):
                prod = 1.0
                for i in range(0, self.M):
                    if self.J[i, j] > 0:
                        x1 = integrate.quad(evalXSin, 0, 1, args=k)[0]
                        prod = prod * special.eval_hermitenorm(int(self.J[i, j]),
                                                               x1) * (1.0 / sqrt(factorial(self.J[i, j])))
                H[j, k] = prod
            x2 = self.x[k]
            H[0, k] = special.eval_hermitenorm(2, x2) - special.eval_hermitenorm(1, x2) + 1
        return H



class set_simulation(integrator):

    def __init__(self, u0, params, JNM):
        super(set_simulation, self).__init__(u0, JNM, **params)


    def SimulaT(self, z, cons):
        """
        Function to simulate Time

        Parameters
        ----------
        tim : array; discretized time
        M : int; number of ODEs
        EigValRe : array, size M
            Array containing the real eigenvalues of matrix A
        EigValIm : array, size M
            Array containing the imaginary eigenvalues of matrix A
        EigVecRe : array, shape(M, M)
            Array containing the real eigenvectors of matrix A
        EigVecIm : array, shape(M, M)
            Array containing the imaginary eigenvectors of matrix A
        cons : array, shape(M, len(xSpace))
            Array containing the ordinary differential equations system constants

        Returns
        -------
        T : array, shape(len(tim), M)
            Array containing the solutions of each ODE for each zi on real space

        """

        T = np.zeros([len(self.t), self.M])
        for k in range(0, len(self.t)):
            for j in range(0, self.M):
                for i in range(0, self.M):
                    if (self.EigValIm[i] != 0 and i == 1) or (i > 1 and self.EigValIm[i] != - self.EigValIm[i - 1]):
                        T[k, j] = T[k, j]\
                                  + cons[i] * exp(self.EigValRe[i] * self.t[k] / pi ** 2) * (
                                self.EigVecRe[j, i] * cos(self.EigValIm[i] * self.t[k] / pi ** 2)
                                - self.EigVecIm[j, i] * sin(self.EigValIm[i] * self.t[k] / pi ** 2)
                                  )
                    elif self.EigValIm[i] == - self.EigValIm[i - 1]:
                        T[k, j] = T[k, j]\
                                  + cons[i] * exp(self.EigValRe[i] * self.t[k] / pi ** 2) * (
                                self.EigVecRe[j, i - 1] * sin(self.EigValIm[i - 1] * self.t[k] / pi ** 2)
                                + self.EigVecIm[j, i - 1] * cos(self.EigValIm[i - 1] * self.t[k] / pi ** 2)
                                  )
                    else:
                        T[k, j] = T[k, j]\
                                  + cons[i] * exp(self.EigValRe[i] * self.t[k] / pi ** 2) * self.EigVecRe[j, i]

        return np.dot(T, self.H1)[:, z]


    def SimulaTX(self):
        """
        Function to simulate Time-X.

        Parameters
        ----------
        xSpace : array; discretized real space

        Returns
        -------
        abs(Tx) : array, shape(len(tim), len(x))
            Array containing the solutions of partial equation

        """
        TX = np.zeros([len(self.t), len(self.x)])
        for zi in range(len(self.x)):
            TX[:, zi] = abs(self.SimulaT(zi, np.dot(np.linalg.inv(self.EigVecRe), self.U_1[:, zi])))

        return TX
