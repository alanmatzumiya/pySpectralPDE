pySpectralPDE: Solver for Partial Differential Equations (PDEs) in its deterministic and stochastic versions. (building)
------------

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`pySpectralPDE` is a Python package for solving the partial differential equations (PDEs) using spectral methods such as
Galerkin and Collocation schemes. This package using different integrator methods to solving in time, for example euler
in its explicit and implicit version, also contains plot tools to built 3D or 2D graphics about solutions.

A first example is the known Burgers' equation in its deterministic and stochastic version. The associated differential operators
are computed using a numba-compiled implementation of spectral methods. This allows defining, inspecting, and solving typical PDEs
that appear for instance in the study of dynamical systems in physics. The focus of the package lies on easy usage to explore the
behavior of PDEs. However, core computations can be compiled transparently using numba for speed.

[Try it out online!](https://mybinder.org/v2/gh/github.com/alanmatzumiya/spectral-methods/PySpectral/master?filepath=examples%2Fjupyter)

Installation
------------

`pySpectralPDE` is available on `pypi`, so you should be able to install it through
`pip`:

```bash
pip install pySpectralPDE
```

Usage
-----

A simple example showing the evolution of the deterministic equation in 1d:

```python
from pySpectralPDE import spectralPDE
import numpy as np


def u0(z):
    return np.exp(- 5.0 ** (-3) * z ** 2)

params = dict(nu=1.0, N=32, xL=-60.0, xR=60.0, dt=0.01, t0=0.0, tmax=100.0)     # setting params
solver = spectralPDE.setup_solver(u0, params)        				# solve the pde
solver.views.plot.graph_3d()                 		     			# plot datas
```
which can be solved for different values of `Diffusion coefficient` in the example above.

This PDE can also be solved in its stochastic version with random forces.
For instance, the [stochastic Burgers' equation](https://en.wikipedia.org/wiki/Burgers'_equation)
can be implemented as
```python
from pySpectralPDE import spectralSPDE
import numpy as np

def u0(x):
    return np.sin(np.pi * x)

   
t = np.linspace(0, 10, 512)
x = np.linspace(0, 1, 256)    
params = {"nu": 0.1, "N": int(5), "M": int(11), "x": x, "t": t}	   # setting params

sol = spectralSPDE.setup_solver(u0=u0, params=params)		   # solve the pde
sol.get_data                                                    
sol.plot                                                           # plot datas
sol.u0_approx
sol.stability
```

More information
----------------
* The [paper published in the Journal](http://dx.doi.org/10.13140/RG.2.2.21593.47203)
