# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Point distributions on the ellipsoid.
"""

import numpy as np
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

        self.__ellipsoid = (a, a*(1-f))
        self.values = np.empty((0, 0))
        self.epoch = None

    def __flattening(self):
        """Compute flattening of the ellipsoid."""
        return 1-self.__ellipsoid[1]/self.__ellipsoid[0]

    def copy(self):
        """Deep copy of GeographicGrid instance."""

        grid = GeographicGrid(a=self.__ellipsoid[0], f=(self.__ellipsoid[0]-self.__ellipsoid[1])/self.__ellipsoid[0])
        grid.lons = self.lons.copy()
        grid.lats = self.lats.copy()
        grid.values = self.values.copy()
        grid.epoch = self.epoch

        return grid

    def radius(self):
        """Geocentric radius of the points along a meridian."""

        f = self.__flattening()
        e2 = 2 * f * (1 - f)
        nu = self.__ellipsoid[0] / np.sqrt(1 - e2 * np.sin(self.lats) ** 2)

        return nu*np.sqrt(np.cos(self.lats)**2 + (1-e2)**2*np.sin(self.lats)**2)

    def colatitude(self):
        """Colatitude of the points along a meridian."""

        f = self.__flattening()
        e2 = 2 * f * (1 - f)
        nu = self.__ellipsoid[0] / np.sqrt(1 - e2 * np.sin(self.lats) ** 2)

        return np.arccos(nu*(1-e2)*np.sin(self.lats)/self.radius())

    def is_regular(self):
        return True


