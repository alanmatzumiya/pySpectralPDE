from numpy import zeros, fft, dot
from numba import jit
from .tools import differentation

class schemes:

    def __init__(self, params, u0, scheme, integrator_method):

        if scheme == 'galerkin':
            ik, k2 = differentation(params['N']).waves_coeff()
            data = zeros([len(params['tdata']) + 1, len(params['x'])], dtype=complex)
            data[0, :] = fft.fft(u0(params['x']))
            if integrator_method == 'explicit':
                self.data = self.explicit(self.galerkin)(data, params['tdata'], params['dt'],
                                                         params['p'], params['nu'], ik, k2)
            else:
                self.data = self.implicit(self.galerkin)(data, params['tdata'], params['dt'],
                                                         params['p'], params['nu'], ik, k2)
        else:
            D1, D2 = differentation(params['N']).D1(), differentation(params['N']).D2()

            data = zeros([len(params['tdata']) + 1, len(params['x'])], dtype=float)
            data[0, :] = u0(params['x'])
            if integrator_method == 'explicit':
                self.data = self.explicit(self.collocation)(data, params['tdata'], params['dt'],
                                                         params['p'], params['nu'], D1, D2)
            else:
                self.data = self.implicit(self.collocation)(data, params['tdata'], params['dt'],
                                                         params['p'], params['nu'], D1, D2)

    @staticmethod
    @jit(target='cpu', nopython=True)
    def galerkin(v_hat, p, nu, ik, k2):

        return p ** 2 * nu * k2 * v_hat - 0.5 * p * ik * v_hat ** 2

    @staticmethod
    @jit(target='cpu', nopython=True)
    def collocation(v, p, nu, D1, D2):

        return p ** 2 * nu * dot(D2, v) - 0.5 * p * dot(D1, v ** 2)

    @staticmethod
    def explicit(f):
        @jit(target='cpu', nopython=True)
        def euler(data, tdata, dt, p, nu, diff_1, diff_2):
            for i in range(1, len(tdata)):
                data[i, :] = data[i - 1, :] + dt * f(data[i - 1, :], p, nu, diff_1, diff_2)
            return data

        return euler

    @staticmethod
    def implicit(f):
        @jit(target='cpu', nopython=True)
        def euler(data, tdata, dt, p, nu, diff_1, diff_2, tol=1.0 ** -10):
            for i in range(1, len(tdata)):
                v_test = data[i - 1, :] + dt * f(data[i - 1, :], p, nu, diff_1, diff_2)
                error = 1.0
                while error > tol:
                    v_new = v_test + dt * f(data[i - 1, :], p, nu, diff_1, diff_2)
                    error = max(abs(v_new - v_test))
                    v_test = v_new
                data[i, :] = v_test
            return data

        return euler

    @staticmethod
    def explicit_multistep(f):
        @jit(target='cpu', nopython=True)
        def RK4(data, tdata, dt, p, nu, diff_1, diff_2):
            for i in range(1, len(tdata)):
                a = f(data[i - 1, :], p, nu, diff_1, diff_2)
                b = f(data[i - 1, :] + 0.5 * a * dt, p, nu, diff_1, diff_2)
                c = f(data[i - 1, :] + 0.5 * b * dt, p, nu, diff_1, diff_2)
                d = f(data[i - 1, :] + c * dt, p, nu, diff_1, diff_2)

                data[i, :] = data[i - 1, :] + dt * (a + 2 * (b + c) + d) / 6.0
            return data

        return RK4
