# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Spatial filters for post-processing of potential coefficients.
"""

from l3py.gravityfield import PotentialCoefficients
import pkg_resources
import numpy as np
import abc


class SpatialFilter(metaclass=abc.ABCMeta):
    """
    Base interface for spatial filters applied to a PotentialCoefficients instance. Derived classes must at least
    implement a `filter` method which takes a PotentialCoefficients instance as argument. The gravity field passed
    to this method should remain unchanged.

    """

    @abc.abstractmethod
    def filter(self, gravityfield):
        pass


class Gaussian(SpatialFilter):
    """
    Implements a Gaussian filter.

    Parameters
    ----------
    radius : float
        filter radius in kilometers
    """

    def __init__(self, radius):

        self.radius = radius

    def filter(self, gravityfield):
        """
        Apply the Gaussian filter to a PotentialCoefficients instance.

        Parameters
        ----------
        gravityfield : PotentialCoefficients instance
            gravity field to be filtered, remains unchanged

        Returns
        -------
        result : PotentialCoefficients instance
            filterd copy of input

        """

        if not isinstance(gravityfield, PotentialCoefficients):
            raise TypeError("Filter operation only implemented for instances of 'PotentialCoefficients'")

        nmax = gravityfield.nmax()

        b = np.log(2.0)/(1-np.cos(self.radius/6378.1366))
        wn = np.zeros(nmax+1)
        wn[0] = 1.0
        wn[1] = (1+np.exp(-2*b))/(1-np.exp(-2*b))-1/b
        for n in range(2, nmax+1):
            wn[n] = -(2*n-1)/b*wn[n-1]+wn[n-2]
            if wn[n] < 1e-7:
                break

        result = gravityfield.copy()
        for n in range(2, nmax+1):
            result.anm[n, 0:n + 1] *= wn[n]
            result.anm[0:n, n] *= wn[n]

        return result


class DDK(SpatialFilter):
    """
    Implements the DDK filter by Kusche et al. (2009).

    Parameters
    ----------
    level : int
        DDK filter level (positive, non-zero)
    """

    def __init__(self, level):

        if level < 1:
            raise ValueError('DDK level must be at least 1 (requested DDK{0:d}).'.format(level))

        normals = np.load(pkg_resources.resource_filename('l3py', 'data/ddk_normals.npz'), allow_pickle=True)['arr_0']
        self.__nmax = normals[0].shape[0]-1
        weights = 10**(15-level) * np.arange(self.__nmax + 1, dtype=float) ** 4
        weights[0] = 1

        self.__array = []
        for normals_block in normals:
            m = self.__nmax + 1 - normals_block.shape[0]
            self.__array.append(np.linalg.solve(normals_block + np.diag(weights[m:]), normals_block))

    def filter(self, gravityfield):
        """
        Apply the DDK filter to a PotentialCoefficients instance.

        Parameters
        ----------
        gravityfield : PotentialCoefficients instance
            gravity field to be filtered, remains unchanged

        Returns
        -------
        result : PotentialCoefficients instance
            filterd copy of input

        Raises
        ------
        ValueError
            if maximum spherical harmonic degree is greater than 120

        """
        if not isinstance(gravityfield, PotentialCoefficients):
            raise TypeError("Filter operation only implemented for instances of 'PotentialCoefficients'")

        nmax = gravityfield.nmax()
        if nmax > self.__nmax:
            raise ValueError('DDK filter only implemented for a maximum degree of {1:d} (nmax={0:d} supplied).'
                             .format(nmax, self.__nmax))

        result = gravityfield.copy()

        result.anm[:, 0] = (self.__array[0][0:nmax+1, 0:nmax+1]@gravityfield.anm[:, 0:1]).flatten()
        for m in range(1, nmax+1):
            result.anm[m::, m] = (self.__array[2*m-1][0:nmax + 1 - m, 0:nmax + 1 - m] @
                                  gravityfield.anm[m::, m:m+1]).flatten()
            result.anm[m-1, m::] = (self.__array[2*m][0:nmax + 1 - m, 0:nmax + 1 - m] @
                                    gravityfield.anm[m-1:m, m::].T).flatten()

        result.anm[0:2, 0:2] = gravityfield.anm[0:2, 0:2].copy()

        return result
