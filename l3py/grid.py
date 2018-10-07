# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.


import numpy as np
import itertools as it
from l3py.gravityfield import WaterHeight
import abc


class Grid(metaclass=abc.ABCMeta):
    """
    Base interface for point collections.

    Subclasses must implement a deep copy, getter for radius and colatitude as well as a method which returns
    whether the grid is regular (e.g. equiangular geographic coordinates) or an arbitrary point distribution.
    """
    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def radius(self):
        pass

    @abc.abstractmethod
    def colatitude(self):
        pass

    @abc.abstractmethod
    def is_regular(self):
        pass


class GeographicGrid(Grid):
    """
    Class representation of a (possibly time stampled) global geographic grid defined by step size in longitude
    and latitude.

    The resulting point coordinates are center points of area elements (pixels). This means that for
    `dlon=dlat=1` the lower left coordinate will be (-179.5, -89.5) and the upper right (179.5, 89.5) degrees.

    Parameters
    ----------
    dlon : float
        longitudinal step size in degrees
    dlat : float
        latitudinal step size in degrees
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid
    """

    def __init__(self, dlon=0.5, dlat=0.5, a=6378137.0, f=298.2572221010**-1):

        nlons = 360 / dlon
        nlats = 180 / dlat

        self.lons = np.linspace(-np.pi+dlon/180*np.pi * 0.5, np.pi-dlon/180*np.pi*0.5, int(nlons))
        self.lats = -np.linspace(-np.pi*0.5 + dlat/180*np.pi * 0.5, np.pi*0.5 - dlat/180*np.pi*0.5, int(nlats))

        e2 = 2*f*(1-f)
        nu = a/np.sqrt(1-e2*np.sin(self.lats)**2)

        self.r = nu*np.sqrt(np.cos(self.lats)**2 + (1-e2)**2*np.sin(self.lats)**2)
        self.theta = np.arccos(nu*(1-e2)*np.sin(self.lats)/self.r)
        self.values = np.empty((0, 0))
        self.epoch = None

    def copy(self):
        """Deep copy of GeographicGrid instance."""

        grid = GeographicGrid()
        grid.lons = self.lons.copy()
        grid.lats = self.lats.copy()
        grid.r = self.r.copy()
        grid.theta = self.theta.copy()
        grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid

    def radius(self):
        """Geocentric radius of the points along a meridian."""

        return self.r

    def colatitude(self):
        """Colatitude of the points along a meridian."""
        return self.theta

    def is_regular(self):
        return True


def legendre_functions(nmax, theta):
    """
    Associated fully normalized Legendre functions (1st kind).

    Parameters
    ----------
    nmax : float
        maximum spherical harmonic degree to compute
    theta : array_like, shape (m,)
        colatitude of evaluation points in radians

    Returns
    -------
    Array containing the fully normalized Legendre functions. Column
    k corresponds to P_nm where k = n*(n+1)/2+m evaluated at all points
    theta.
    """
    P = np.zeros((theta.size, int((nmax + 1) * (nmax + 2) / 2)))

    P[:, 0] = 1.0  # initial values for recursion
    P[:, 1] = np.sqrt(3) * np.cos(theta)
    P[:, 2] = np.sqrt(3) * np.sin(theta)

    for m, n in it.combinations_with_replacement(np.arange(nmax + 1), 2):

        col = int(n * (n + 1) * 0.5 + m)  # degree wise ordering
        if col < 3:  # we start recursion at P_20
            continue

        if m == n:
            P[:, col] = np.sqrt((2.0 * n + 1.0) / (2.0 * n)) * np.sin(theta) * P[:, col - n - 1]

        elif m + 1 == n:
            P[:, col] = np.sqrt(2.0 * n + 1.0) * np.cos(theta) * P[:, col - n]

        else:
            P[:, col] = np.sqrt((2.0 * n - 1.0) / (n - m) * (2.0 * n + 1.0) / (n + m)) * \
                        np.cos(theta) * P[:, col - n] - \
                        np.sqrt((2.0 * n + 1.0) / (2.0 * n - 3.0) * (n - m - 1.0) / (n - m) * (n + m - 1.0) / (n + m)) \
                        * P[:, col - 2 * n + 1]

    return P


def shc2grid(gravityfield, grid=GeographicGrid(), kernel='ewh'):
    """

    """
    if kernel in ('ewh',):
        inverse_coefficients = WaterHeight(gravityfield.nmax())
    else:
        raise ValueError("Unrecognized kernel ({0:s})".format(kernel))

    if grid.is_regular():
        P = legendre_functions(gravityfield.nmax(), grid.colatitude())
        Rr = (gravityfield.R / grid.r)

        gridded_values = np.zeros((grid.lats.size, grid.lons.size))

        for n in range(gravityfield.nmax()+1):
            coeffs = gravityfield.by_degree(n)
            orders = [c.m for c in coeffs]
            idx = [int(n*(n+1)*0.5+m) for m in orders]

            kn = inverse_coefficients.kn(n, grid.r, grid.lats)

            CS = np.vstack([c.trigonometric_function(c.m*grid.lons)*c.value for c in coeffs])
            gridded_values += (kn*Rr**(n+1))[:, np.newaxis]*P[:, idx]@CS

        output_grid = grid.copy()
        output_grid.values = gridded_values
        output_grid.epoch = gravityfield.epoch
    else:
        raise NotImplementedError('Propagation to arbitrary point distributions is not yet implemented.')

    return output_grid
