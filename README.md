
l3py - L3 Products in Python
============================

What is l3py
------------

l3py is a free Open Source software package for computing gridded mass anomalies (Level 3 products)
from time variable gravity field solutions. It is tailored for products of the GRACE
mission and its successor GRACE-FO.

The features of l3py are:

 * File I/O for common data formats including GFC files and COARDS compliant netCDF
 * Basic arithmetic operations for sets of potential coefficients
 * Propagation of spherical harmonic coefficients to gridded mass anomalies
 * Spatial filtering of potential coefficients

Installation
------------

l3py is written for Python>=3.4 and only depends on `numpy` and `netcdf4`. The current release is hosted 
on the PyPI and can be installed through:

    pip install l3py
    
To install the current development version of the package, first clone the repository or download the zip archive. 
In the root directory
of the package (i.e. the directory containing the ``setup.py`` file), running

    pip install .

will install the package and its dependencies.

API Reference
-------------

A complete API reference of the module is available on readthedocs: https://l3py.readthedocs.io/en/latest.

[![Documentation Status](https://readthedocs.org/projects/l3py/badge/?version=latest)](https://l3py.readthedocs.io/en/latest/?badge=latest)


License
-------

l3py a free Open Source software released under the MIT license.

If you wish to cite l3py in a publication, please use the following DOI.

[![DOI](https://zenodo.org/badge/151739427.svg)](https://zenodo.org/badge/latestdoi/151739427)
