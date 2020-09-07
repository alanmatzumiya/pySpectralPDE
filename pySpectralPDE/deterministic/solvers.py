from numpy import zeros, fft, dot, argmax, real
from numba import jit
from .tools import differentation

class schemes:

    def __init__(self, params, u0, scheme, integrator_method):

        if scheme == 'galerkin':
            ik, k2 = differentation(params['N']).waves_coeff()
            data = zeros([len(params['tdata']), len(params['x'])], dtype=complex)
            data[0, :] = u0(params['x'])
            if integrator_method == 'explicit':
                self.data = self.galerkin_explicit(self.galerkin, params['tdata'], data, params['dt'],
                                                         params['p'], params['nu'], ik, k2)
            else:
                self.data = self.implicit(self.galerkin)(data, params['tdata'], params['dt'],
                                                         params['p'], params['nu'], ik, k2)
        else:
            D1, D2 = differentation(params['N']).D1(), differentation(params['N']).D2()

            data = zeros([len(params['tdata']), len(params['x'])], dtype=float)
            data[0, :] = u0(params['x'])
            if integrator_method == 'explicit':
                self.data = self.collocation_explicit(self.collocation)(data, params['tdata'], params['dt'],
                                                         params['p'], params['nu'], D1, D2)
            else:
                self.data = self.implicit(self.collocation)(data, params['tdata'], params['dt'],
                                                         params['p'], params['nu'], D1, D2)


    @staticmethod
    @jit(target='cpu', nopython=True)
    def galerkin(v_hat, p, nu, ik, k2):

        return - p ** 2 * nu * k2 * v_hat, - p * ik * v_hat

    @staticmethod
    @jit(target='cpu', nopython=True)
    def collocation(v, p, nu, D1, D2):

        return p ** 2 * nu * dot(D2, v) - 0.5 * p * dot(D1, v ** 2)

    @staticmethod
    def galerkin_explicit(f, tdata, data, dt, p, nu, diff_1, diff_2):
        for i in range(1, len(tdata)):
            F = f(fft.fft(data[i - 1, :]), p, nu, diff_1, diff_2)
            data[i, :] = data[i - 1, :] + dt * (
                    real(fft.ifft(F[0])) + data[i - 1, :] * real(fft.ifft(F[1]))
            )
        return data

    @staticmethod
    def collocation_explicit(f):
        @jit(target='cpu', nopython=True)
        def euler(data, tdata, dt, p, nu, diff_1, diff_2):
            for i in range(1, len(tdata)):
                data[i, :] = data[i - 1, :] + dt * f(data[i - 1, :], p, nu, diff_1, diff_2)
            return data
        return euler

    @staticmethod
    def implicit(f):
        @jit(target='cpu', nopython=True)
        def euler(data, tdata, dt, p, nu, diff_1, diff_2):
            for i in range(1, len(tdata)):
                v_test = data[i - 1, :] + dt * f(data[i - 1, :], p, nu, diff_1, diff_2)
                while argmax(v_test - data[i, :]) > 1.0 * 10 ** (-8):
                    v_test = data[i-1, :] + dt * f(v_test, p, nu, diff_1, diff_2)

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
