# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

import numpy as np
import pkg_resources
import abc
import l3py.utilities


class Kernel(metaclass=abc.ABCMeta):
    """
    Base interface for spherical harmonic kernels.

    Subclasses must implement a method `kn` which depends on degree radius and co-latitude and returns kernel
    coefficients.
    """

    @abc.abstractmethod
    def kn(self, r, colat):
        pass


class WaterHeight(Kernel):
    """
    Implementation of the water height kernel. Applied to a sequence of potential coefficients, the result is
    equivalent water height in meters when propagated to space domain.

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

    def kn(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """
        return self.__kn[n]/r


class OceanBottomPressure(Kernel):
    """
        Implementation of the ocean bottom pressure kernel. Applied to a sequence of potential coefficients, the result
        is ocean bottom pressure in Pascal when propagated to space domain.

        Parameters
        ----------
        nmax : int
            maximum spherical harmonic degree
        rho : float
            density of water in [kg/m**3]
        """

    def __init__(self, nmax):
        file_name = pkg_resources.resource_filename('l3py', 'data/loadLoveNumbers_Gegout97.txt')
        love_numbers = np.loadtxt(file_name)
        love_numbers.resize((nmax + 1,))

        self.__kn = (2 * np.arange(0, nmax + 1) + 1) / (1 + love_numbers[0:nmax + 1]) / (4 * np.pi * 6.673e-11)

    def kn(self, n, r=6378136.6, colat=0):
        """
        Kernel coefficient for degree n.

        Parameters
        ----------
        n : int
            coefficient degree
        r : float, array_like shape (m,)
            radius of evaluation points
        colat : float, array_like shape (m,)
            co-latitude of evaluation points in radians

        Returns
        -------
        kn : float, array_like shape (m,)
            kernel coefficients for degree n for all evaluation points
        """

        return self.__kn[n]/r*l3py.utilities.normal_gravity(r, colat)


class SurfaceDensity(Kernel):
    pass


