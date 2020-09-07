"""
Solvers define how a pde is solved, i.e., advanced in time.
.. autosummary::
.. codeauthor:: Alan Matzumiya <alan.matzumiya@gmail.com>
"""

from typing import List

from .setup_solver import FPK_solver

__all__ = [
    "FPK_solver"
]