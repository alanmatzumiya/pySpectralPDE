from numpy import pi, linspace, array, round, arange, zeros, fft, real, dot
from .solvers import integrator
from .helpers import eval_aprox
from .grapher import graph_defaults
from .tools import *


class setup_solver:

    def __init__(self, u0, params, scheme='collocation', integrator_method='explicit'):

        params['h'] = 2.0 * pi / params['N']
        params['p'] = 2.0 * pi / (params['xR'] - params['xL'])

        x = array(arange(- params['N'] / 2, params['N'] / 2, 1), dtype=float)
        params['x'] = x * params['h'] / params['p']

        operator_diff = differentation(params['N'])
        ik = lambda v: v * real(fft.ifft(params['p'] * operator_diff.ik * fft.fft(v)))
        k2 = lambda v: real(fft.ifft(params['nu'] * (operator_diff.ik * params['p']) ** 2 * fft.fft(v)))
        diff1 = lambda v: params['p'] * v * dot(operator_diff.diff1, v)
        diff2 = lambda v: params['nu'] * dot(operator_diff.diff2, v) * params['p'] ** 2

        discrete_form = {"galerkin": lambda v: - ik(v) + k2(v),
                         "collocation": lambda v: - diff1(v) + diff2(v)}

        time_integrator = integrator()
        method = {"explicit": time_integrator.explicit, "implicit": time_integrator.implicit}

        params['t'], params['data'] = method[integrator_method](discrete_form[scheme],
                                                   data_PVI=[params['x'], u0(params['x']), params['t0'], params['tmax'], params['dt']]
                                                   )
        self.params = params
        self.views = graph_defaults()
        self.scheme = scheme

    def __call__(self, x, tdata):

        fourier = eval_aprox()
        expansion_evaluator = {'galerkin': fourier.continuous_expansion, 'collocation': fourier.discrete_expansion}
        return expansion_evaluator[self.scheme](x, tdata, self.params['N'])
