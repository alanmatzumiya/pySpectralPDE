PySpectral: Solver for Burgers' Equation (building)
------------

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`PySpectral` is a Python package for solving the partial differential equation (PDE)
of Burgers' equation in its deterministic and stochastic version. The associated differential operators are computed using a
numba-compiled implementation of spectral methods. This allows defining,
inspecting, and solving typical PDEs that appear for instance in the study of
dynamical systems in physics. The focus of the package lies on easy usage to
explore the behavior of PDEs. However, core computations can be compiled
transparently using numba for speed.

[Try it out online!](https://mybinder.org/v2/gh/github.com/alanmatzumiya/Spectral-Methods/burgers/master?filepath=examples%2Fjupyter)

Installation
------------

`PySpectral` is available on `pypi`, so you should be able to install it through
`pip`:

```bash
pip install PySpectral
```

Usage
-----

A simple example showing the evolution of the deterministic equation in 1d:

```python
from PySpectral import deterministic_solver

result = deterministic_solver()         # solve the pde
result.plot.graph_3d()                  # plot the resulting field
```
which can be solved for different values of `Diffusion coefficient` in the example above.

This PDE can also be solved in its stochastic version with random forces.
For instance, the [stochastic Burgers' equation](https://en.wikipedia.org/wiki/Burgers'_equation)
can be implemented as
```python
from PySpectral import stochastic_solver

result = stochastic_solver()          # solve the pde
result.plot.graph_3d()                # plot the resulting field
```

More information
----------------
* The [paper published in the Journal](http://dx.doi.org/10.13140/RG.2.2.21593.47203)