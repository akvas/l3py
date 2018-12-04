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
        self.__area = 2.0*dlon/180*np.pi*np.sin(dlat*0.5/180*np.pi)*np.cos(self.lats)

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

    def mean(self, mask=None):
        """
        Compute the weighted average of grid points, potentially with a mask. The individual points are weighted
        by their area elements.

        Parameters
        ----------
        mask : array_like(nlons, nlats), None
            boolean array with the same shape as the value array. If None, all points are averaged.

        Returns
        -------
        mean : float
            weighted mean over all grid points in mask

        See Also
        --------
        l3py.grid.GeographicGrid.create_mask : member function which creates masks from polygons

        """
        if mask is None:
            mask = np.ones((self.lats.size, self.lons.size), dtype=bool)

        areas = np.tile(self.__area[:, np.newaxis], (1, self.lons.size))[mask]

        return np.sum(areas*self.values[mask])/np.sum(areas)

    def create_mask(self, polygon):
        """
        Create a mask (boolean array) for the Geographic grid instance based on a polygon.

        Parameters
        ----------
        polygon : Polygon
            Polygon instance. This method also supports shapely Polygons through duck typing.

        Returns
        -------
        mask : array_like(m,n)
            boolean array of size(nlons, nlats), True for points inside the polygon, False for points outside.
        """

        lons, lats = np.meshgrid(self.lons*180/np.pi, self.lats*180/np.pi)

        mask = winding_number(polygon.exterior.coords, lons, lats)
        for interior in polygon.interiors:
            mask = np.logical_and(mask, ~winding_number(interior.coords, lons, lats))

        return mask


class Polygon:
    """
    Simple class representation of a Polygon, potentially with holes. No sanity checking for potential geometry errors
    is performed.

    Parameters
    ----------
    point_list : list of (lon,lat) tuples
        Point list defining the polygon. The (lon,lat) tuples should be given in degrees.
    holes : list of list of (lon,lat) tuples
        List of point lists defining holes within the polygon. The (lon,lat) tuples should be given in degrees.
    """
    def __init__(self, point_list, holes=None):

        self.exterior = LinearRing(point_list)
        if holes is not None:
            self.interiors = [LinearRing(h) for h in holes]
        else:
            self.interiors = []


class LinearRing:
    """
    Class representation of a point list.

    Parameters
    ----------
    point_list : list of (lon,lat) tuples
       Point list defining the polygon. The (lon,lat) tuples should be given in degrees.
    """
    def __init__(self, point_list):

        self.coords = list(point_list)
        if self.coords[0] != self.coords[-1]:
            self.coords.append(self.coords[0])


def winding_number(point_list, x, y):
    """
    Winding number algorithm for point in polygon tests.

    Parameters
    ----------
    point_list : list of (x,y) tuples
        point list defining the polygon
    x : ndarray(m,), ndarray(m,n)
        x-coordinates of points to be tested
    y : ndarray(m,), ndarray(m,n)
        y-coordinates of points to be tested

    Returns
    -------
    contains : ndarray(m,), ndarray(m,n)
        boolean array indicating which point is contained in the polygon
    """
    coords = list(point_list)
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    wn = np.zeros(x.shape, dtype=int)

    for p0, p1 in zip(coords[0:-1], coords[1:]):
        l1 = p0[1] <= y
        l2 = p1[1] > y

        loc_to_edge = (p1[0] - p0[0]) * (y - p0[1]) - (x - p0[0]) * (p1[1] - p0[1])

        wn[np.logical_and(np.logical_and(l1, l2), loc_to_edge > 0)] += 1
        wn[np.logical_and(np.logical_and(~l1, ~l2), loc_to_edge < 0)] -= 1

    return wn != 0
