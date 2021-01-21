from numpy import argmax, round, linspace, zeros


class integrator:

    @staticmethod
    def explicit(f, data_PVI):
        
        x, v0, t0, tmax, dt = data_PVI
        t = linspace(
        t0, tmax, int(round(tmax / dt)) + 1,
            dtype=float)
        nplots = int(round(tmax / 0.5))
        data = zeros([len(t), len(x)])
        data[0, :] = v0
        for i in range(1, len(t)):
                try:
                    data[i, :] = data[i - 1, :] + dt * f(data[i - 1, :])
                except RuntimeError:
                    break
        return t[::nplots], data[::nplots, : ]

    @staticmethod
    def implicit(f, data_PVI):
        x, v0, t0, tmax, dt = data_PVI
        t = linspace(
        t0, tmax, int(round(tmax / dt)) + 1,
            dtype=float)
        nplots = int(round(tmax / 0.5))
        data = zeros([len(t), len(x)])
        data[0, :] = v0
        for i in range(1, len(t)):
            test = data[i - 1, :] + dt * f(data[i - 1, :])
            while argmax(test - data[i, :]) > 1.0 * 10 ** (-8):
                test = data[i-1, :] + dt * f(test)
            data[i, :] = test
        return t[::nplots], data[::nplots, : ]

    @staticmethod
    def explicit_multistep(f, data_PVI):
        x, v0, t0, tmax, dt = data_PVI
        t = linspace(
        t0, tmax, int(round(tmax / dt)) + 1,
            dtype=float)
        nplots = int(round(tmax / 0.5))
        data = zeros([len(t), len(x)])
        data[0, :] = v0
        for i in range(1, len(t)):
            a = f(data[i - 1, :])
            b = f(data[i - 1, :] + 0.5 * a * dt)
            c = f(data[i - 1, :] + 0.5 * b * dt)
            d = f(data[i - 1, :] + c * dt)
            data[i, :] = data[i - 1, :] + dt * (a + 2 * (b + c) + d) / 6.0
        return t[::nplots], data[::nplots, : ]
