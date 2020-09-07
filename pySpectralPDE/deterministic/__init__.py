"""
Solvers define how a pde is solved, i.e., advanced in time.
.. autosummary::
.. codeauthor:: Alan Matzumiya <alan.matzumiya@gmail.com>
"""

from typing import List
from .setup_solver import deterministic_solver

__all__ = [
    "spectral_solver"
]
