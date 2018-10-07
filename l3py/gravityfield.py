# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.


import numpy as np
import pkg_resources
import abc


class Coefficient:
    """
    Class representation of a spherical harmonic coefficient.

    Parameters
    ----------
    trigfun : numpy sine or cosine
        trigonometric function associated with the coefficient
    n : int
        spherical harmonic degree
    m : int
        spherical harmonic order
    value : float
        numeric value associated with the coefficient
    """

    def __init__(self, trigfun, n, m, value=0.0):
        self.n = n
        self.m = m
        self.value = value
        self.trigonometric_function = trigfun


class GravityField:
    """
    Class representation of a set of (possibly time stamped) potential coefficients.

    Parameters
    ----------
    GM : float
        geocentric gravitational constant
    R : float
        reference radius
    """
    def __init__(self, GM=3.9860044150e+14, R=6.3781363000e+06):

        self.GM = GM
        self.R = R
        self.anm = np.zeros((0, 0))
        self.epoch = None

    def copy(self):
        """Return a deep copy of the gravity field instance"""
        gf = GravityField(self.GM, self.R)
        gf.anm = self.anm.copy()

        return gf

    def append(self, *coeffs):
        """Append a coefficient to a gravity field."""
        for coeff in coeffs:
            if coeff.n > self.nmax():
                tmp = np.zeros((coeff.n+1, coeff.n+1))
                tmp[0:self.anm.shape[0], 0:self.anm.shape[1]] = self.anm.copy()
                self.anm = tmp

            if coeff.trigonometric_function == np.cos:
                self.anm[coeff.n, coeff.m] = coeff.value * self.GM / self.R
            elif coeff.trigonometric_function == np.sin and coeff.m > 0:
                self.anm[coeff.m-1, coeff.n] = coeff.value * self.GM / self.R

    def truncate(self, nmax):
        """Truncate a gravity field to a new maximum spherical harmonic degree."""
        if nmax < self.nmax():
            self.anm.resize((nmax+1, nmax+1))

    def replace_c20(self, gravityfield):
        """
        Replace the c20 coefficient of a gravity field by c20 of another gravity field. Substitution is performed
        in-place.

        Parameters
        ----------
        gravityfield : GravityField instance
            gravity field containing the new c20 value

        """
        self.anm[2, 0] = gravityfield.anm[2, 0]

    def by_degree(self, n):
        """
        Return all coefficients of a specific spherical harmonic degree n

        Parameters
        ----------
        n : int
            spherical harmonic degree

        Returns
        -------
        coeffs : list of Coefficient instances
            all coefficients associated with degree n
        """
        coeffs = []
        for m in range(0, n+1):
            coeffs.append(Coefficient(np.cos, n, m, self.anm[n, m]))
        for m in range(1, n + 1):
            coeffs.append(Coefficient(np.sin, n, m, self.anm[m-1, n]))

        return coeffs

    def nmax(self):
        """Return maximum spherical harmonic degree of gravity field."""
        return self.anm.shape[0]-1

    def __add__(self, other):
        """Coefficient-wise addition of two gravity fields."""
        if not isinstance(other, GravityField):
            raise TypeError("unsupported operand type(s) for +: 'Gravityfield' and '"+type(other)+"'")

        if self.nmax() > other.nmax():
            result = self.copy()
            result.anm[0:other.anm.shape[0], 0:other.anm.shape[1]] += other.anm
        else:
            result = other.copy()
            result.anm[0:self.anm.shape[0], 0:self.anm.shape[1]] += self.anm

        return result

    def __sub__(self, other):
        """Coefficient-wise subtraction of two gravity fields."""
        if not isinstance(other, GravityField):
            raise TypeError("unsupported operand type(s) for -: 'Gravityfield' and '"+type(other)+"'")

        return self+(other*-1)

    def __mul__(self, other):
        """Multiplication of a gravity field with a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for *: 'Gravityfield' and '"+type(other)+"'")

        result = self.copy()
        result.anm *= other

        return result

    def __truediv__(self, other):
        """Division of a gravity field by a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for /: 'Gravityfield' and '"+type(other)+"'")

        return self*(1.0/other)


class Kernel(metaclass=abc.ABCMeta):
    """
    Base interface for spherical harmonic kernels.

    Subclasses must implement a method `kn` which depends on degree radius and latitude and returns kernel
    coefficients.
    """

    @abc.abstractmethod
    def kn(self, r, lat):
        pass


class WaterHeight(Kernel):
    """
    Implementation of the water height kernel. Applied to a sequence of potential coefficients, the result is
    equivalent water height when propagated to space domain.

    Parameters
    ----------
    nmax : int
        maximum spherical harmonic degree
    rho : float
        density of water in [kg/m**3]
    """
    def __init__(self, nmax, rho=1025):

        file_name = pkg_resources.resource_filename('l3py', 'data/loadLoveNumbers_Gegout97.txt')
        love_numbers = np.loadtxt(file_name)
        love_numbers.resize((nmax+1,))

        self.__kn = (2*np.arange(0, nmax+1)+1)/(1+love_numbers[0:nmax+1])/(4*np.pi*6.673e-11*rho)

    def kn(self, n, r=6378136.6, lat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        dat : float, array_like shape (m,)
            latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """
        return self.__kn[n]/r
