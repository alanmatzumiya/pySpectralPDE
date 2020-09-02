"""
The py-burgers package provides classes and methods for solving burgers' equation
in its deterministic and stochastic version using spectral methods.
.. autosummary::
.. codeauthor:: Alan Matzumiya <alan.matzumiya@gmail.com>
"""

from typing import List
from .burgers.deterministic.setup_solver import deterministic_solver
from .burgers.stochastic.setup_solver import stochastic_solver
from .version import __version__

__all__ = [
    "deterministic_solver",
    "stochastic_solver",
    "__version__"
]