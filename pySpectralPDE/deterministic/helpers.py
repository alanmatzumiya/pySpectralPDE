from numpy import zeros, exp, sqrt, pi, arange, allclose, array, polynomial
from scipy import optimize
from scipy.integrate import trapz, odeint
from scipy.optimize import curve_fit
from numba import jit


class analytic_solution:

    def analytical_solution(self, NT, NX, TMAX, XMAX, NU):
        """
        Returns the velocity field and distance for the analytical solution
        """

        # Increments
        DT = TMAX / (NT - 1)
        DX = XMAX / (NX - 1)

        # Initialise data structures
        import numpy as np
        u_analytical = np.zeros((NX, NT))
        x = np.zeros(NX)
        t = np.zeros(NT)

        # Distance
        for i in range(0, NX):
            x[i] = i * DX

        # Analytical Solution
        for n in range(0, NT):
            t = n * DT

            for i in range(0, NX):
                phi = exp(-(x[i] - 4 * t) ** 2 / (4 * NU * (t + 1))) + exp(
                    -(x[i] - 4 * t - 2 * PI) ** 2 / (4 * NU * (t + 1)))

                dphi = (-0.5 * (x[i] - 4 * t) / (NU * (t + 1)) * exp(-(x[i] - 4 * t) ** 2 / (4 * NU * (t + 1)))
                        - 0.5 * (x[i] - 4 * t - 2 * PI) / (NU * (t + 1)) * exp(
                            -(x[i] - 4 * t - 2 * PI) ** 2 / (4 * NU * (t + 1))))

                u_analytical[i, n] = -2 * NU * (dphi / phi) + 4

        return u_analytical, x

    def convection_diffusion(self, NT, NX, TMAX, XMAX, NU):
        """
        Returns the velocity field and distance for 1D non-linear convection-diffusion
        """

        # Increments
        DT = TMAX / (NT - 1)
        DX = XMAX / (NX - 1)

        # Initialise data structures
        import numpy as np
        u = np.zeros((NX, NT))
        u_analytical = np.zeros((NX, NT))
        x = np.zeros(NX)
        t = np.zeros(NT)
        ipos = np.zeros(NX)
        ineg = np.zeros(NX)

        # Periodic boundary conditions
        for i in range(0, NX):
            x[i] = i * DX
            ipos[i] = i + 1
            ineg[i] = i - 1

        ipos[NX - 1] = 0
        ineg[0] = NX - 1

        # Initial conditions
        for i in range(0, NX):
            phi = exp(-(x[i] ** 2) / (4 * NU)) + exp(-(x[i] - 2 * PI) ** 2 / (4 * NU))
            dphi = -(0.5 * x[i] / NU) * exp(-(x[i] ** 2) / (4 * NU)) - (0.5 * (x[i] - 2 * PI) / NU) * exp(
                -(x[i] - 2 * PI) ** 2 / (4 * NU))
            u[i, 0] = -2 * NU * (dphi / phi) + 4

        # Numerical solution
        for n in range(0, NT - 1):
            for i in range(0, NX):
                u[i, n + 1] = (u[i, n] - u[i, n] * (DT / DX) * (u[i, n] - u[ineg[i], n]) +
                               NU * (DT / DX ** 2) * (u[ipos[i], n] - 2 * u[i, n] + u[ineg[i], n]))

        return u, x

    def inviscid_solution(self, u0, space, time):

        def F(z, space, time):
            return z + u0(z) * time - space
        exact = zeros([len(time), len(space)])
        exact[0, :] = u0(space)
        for i in range(1, len(time)):
            Z = zeros(len(space))
            for j in range(len(space)):
                zj = optimize.root(F, array(0.0), args=(space[j], time[i]), tol=10 ** -10)
                Z[j] = zj.x
            exact[i, :] = u0(Z)

        return exact

    def exact(self, u0, nu, x, t):

        integ_top = lambda z, xi, tj, nu: (xi - z) * exp(- 2.0 * nu * (xi - z) ** 2) / (4.0 * nu * tj)
        integ_bottom = lambda z, xi, tj, nu: exp(- 2.0 * nu * (xi - z) ** 2) / (4.0 * nu * tj)
        LX = len(x)
        if isinstance(t, float):
            t = [0.0, t]
        LT = len(t)
        TX = zeros([LT, LX], dtype=float)
        TX[0, :] = u0(x)
        for j in range(1, len(t)):
            for i in range(0, len(x)):
                try:
                    TX[j, i] = abs(
                        integ_top(x[i], t[j], nu, j) / integ_bottom(x[i], t[j], nu, j)
                    ) / t[j]

                except RuntimeWarning and ZeroDivisionError:
                    TX[j, i] = 0.0
        return TX


    @staticmethod
    def system(v, p, nu, diff_1, diff_2):

        return p ** 2 * nu * diff_1(v) - 0.5 * p * diff_2(v)

    @staticmethod
    def numerical_solution(df, v0, tdata, p, nu, diff_1, diff_2):
        @jit(target='cpu', nopython=True)
        def solver():
            return odeint(df, v0, tdata, args=[p, nu, diff_1, diff_2], rtol=1.49012e-16)
        return solver

    def kernel_gauss(self, x, t, alpha):

        return exp(- x ** 2 / 4.0 * alpha * t)

    def viscid_solution_1(self, u0, x, t, nu):
        exact = zeros([len(t), len(x)]); exact[0, :] = u0(x)
        rule1 = polynomial.hermite_e.hermegauss(100)
        rulesX = rule1[0][::-1]; rulesW = rule1[1]
        for j in range(0, len(x)):
            for i in range(1, len(t)):
                factor = sqrt(2.0 * nu * t[i])
                z = x[j] - rulesX * factor
                sum1 = 0.0
                for k in range(len(z)):
                    sum1 = sum1 + factor * u0(z[k]) * rulesW[k]
                exact[i, j] = (1.0 / sqrt(4.0 * nu * t[i] * pi)) * sum1
        return exact

    def viscid_solution_2(self, u0, x, t, nu):
        y = arange(-200, 200 + 1, 400)
        data = zeros([len(t), len(x)])
        data[0, :] = u0(x)
        for j in range(len(x)):
            for i in range(1, len(t)):
                sum1 = 0.0
                for k in range(1, len(y) - 1):
                    sum1 = sum1 + self.kernel_gauss(x[j] - y[k], t[i], nu) * u0(y[k])
                G_left = self.kernel_gauss(x[j] - y[0], t[i], nu) * u0(y[0])
                G_right = self.kernel_gauss(x[j] - y[len(y) - 1], t[i], nu) * u0(y[len(y) - 1])
                data[i, j] = (1.0 / sqrt(4.0 * nu * t[i] * pi)) * (sum1 + 0.5 * (G_right - G_left))
        return data


class eval_aprox:

    @staticmethod
    def continuous_expansion(x, v_hat, N):
        k = arange(-int(N/2), int(N/2), 1)
        data = zeros(len(x))
        for i in range(0, len(array(x))):
            data[i] = sum(v_hat * exp(1j * k * x[i])).real / N
        return data

    @staticmethod
    def discrete_expansion(x, v, N):
        k = arange(-int(N / 2), int(N / 2), 1)
        h = 2.0 * pi / N
        data = zeros(len(x))
        for i in range(len(array(x))):
            if allclose(h * i, x[i]):
                data[i] = v[i]
            else:
                data[i] = sum(v * exp(1j * k * x[i])).real / N
        return data


class analysis:

    @staticmethod
    def tester(x, y):
        func = lambda x, a, b: a * exp(- b * x)

        return curve_fit(func, x, y, p0=(1.0, 0.001))

    @staticmethod
    def distance(aprox, exact, space):

        return sqrt(trapz(abs(aprox - exact) ** 2, space)), max(abs(aprox - exact))


