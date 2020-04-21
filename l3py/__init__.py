# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

"""
l3py - L3 Products in Python
============================

What is l3py
------------

l3py is a free Open Source software package for computing mass anomalies from
time variable gravity field solutions. It is tailored for products of the GRACE
mission and its successor GRACE-FO.

The features of l3py are:

 * File I/O for common data formats including GFC files and COARDS compliant netCDF
 * Basic arithmetic operations for sets of potential coefficients
 * Propagation of spherical harmonic coefficients to gridded mass anomalies
 * Spatial filtering of potential coefficients

Modules
-------

.. autosummary::
    :toctree: _generated
    :template: l3py_module.rst

    l3py.io
    l3py.grid
    l3py.gravityfield
    l3py.filter
    l3py.utilities
    l3py.kernel
    l3py.time



"""

from . import io
from . import grid
from . import gravityfield
from . import filter
from . import utilities
from . import kernel
from . import time

__all__ = ["io", "grid", "gravityfield", "filter", "kernel", "utilities", "time"]
