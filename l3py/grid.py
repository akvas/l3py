# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Point distributions on the ellipsoid.
"""

import numpy as np
import abc
import l3py.utilities


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
        return l3py.utilities.geocentric_radius(self.lats, self.__ellipsoid[0], self.__flattening())

    def colatitude(self):
        """Colatitude of the points along a meridian."""
        return l3py.utilities.colatitude(self.lats, self.__ellipsoid[0], self.__flattening())

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

    def create_mask(self, basin):
        """
        Create a mask (boolean array) for the Geographic grid instance based on a polygon.

        Parameters
        ----------
        basin : Basin
            Basin instance.

        Returns
        -------
        mask : array_like(m,n)
            boolean array of size(nlons, nlats), True for points inside the polygon, False for points outside.
        """

        lons, lats = np.meshgrid(self.lons, self.lats)

        return basin.contains_points(lons, lats)


class Basin:
    """
    Simple class representation of an area enclosed by a polygon boundary, potentially with holes. No sanity checking
    for potential geometry errors is performed.

    Parameters
    ----------
    polygons : ndarray(k, 2) or  list of ndarray(k, 2)
        Coordinates defining the basin. Can be either a single two-column ndarray with longitude/latitude pairs for
        rows, or a list of ndarrays in the same format. Longitude/latitude should be given in radians.
    """
    def __init__(self, polygons):

        if isinstance(polygons, np.ndarray):
            self.__polygons = polygons,
        else:
            self.__polygons = polygons

    def contains_points(self, lon, lat):
        """
        Method to check whether points are within the basin bounds.

        Parameters
        ----------
        lon : float, ndarray(m,), ndarray(m,n)
            longitude of points to be tested (should be given in radians)
        lat : float, ndarray(m,), ndarray(m,n)
            latitude of points to be tested (should be given in radians)
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)

        wn = np.zeros(lon.shape if lat.size == 1 else lat.shape, dtype=int)

        for polygon in self.__polygons:
            wn += winding_number(polygon, lon, lat)

        return np.mod(wn, 2).astype(bool)


def winding_number(polygon, x, y):
    """
    Winding number algorithm for point in polygon tests.

    Parameters
    ----------
    polygon : ndarray(k, 2)
        two-column ndarray with longitude/latitude pairs defining the polygon
    x : ndarray(m,), ndarray(m,n)
        x-coordinates of points to be tested
    y : ndarray(m,), ndarray(m,n)
        y-coordinates of points to be tested

    Returns
    -------
    contains : ndarray(m,), ndarray(m,n)
        boolean array indicating which point is contained in the polygon
    """
    coords = polygon
    if np.any(polygon[0] != polygon[-1]):
        coords = np.append(polygon, polygon[0][np.newaxis, :], axis=0)

    wn = np.zeros(x.shape if y.size == 1 else y.shape, dtype=int)

    for p0, p1 in zip(coords[0:-1], coords[1:]):
        l1 = p0[1] <= y
        l2 = p1[1] > y

        loc_to_edge = (p1[0] - p0[0]) * (y - p0[1]) - (x - p0[0]) * (p1[1] - p0[1])

        wn[np.logical_and(np.logical_and(l1, l2), loc_to_edge > 0)] += 1
        wn[np.logical_and(np.logical_and(~l1, ~l2), loc_to_edge < 0)] -= 1

    return wn != 0


def spherical_distance(lon1, lat1, lon2, lat2, r=6378136.3):
    """
    Compute the spherical distance between points (lon1, lat1) and (lon2, lat2) on a sphere with
    radius r.

    Parameters
    ----------
    lon1 : float, array_like(m,), array_like(m,n)
        longitude of source points in radians
    lat1 : float, array_like(m,), array_like(m,n)
        latitude of source points in radians
    lon2 : float, array_like(m,), array_like(m,n)
        longitude of target points in radians
    lat2 : float, array_like(m,), array_like(m,n)
        latitude of target points in radians
    r : float
        radius of the sphere in meters

    Returns
    -------
    d : ndarray(m,), ndarray(m,n)
        spherical distance between points (lon1, lat1) and (lon2, lat2) in meters
    """
    return r*np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))


def ellipsoidal_distance(lon1, lat1, lon2, lat2, a=6378137.0, f=298.2572221010**-1):
    """
    Compute the distance between points (lon1, lat1) and (lon2, lat2) on an ellipsoid with
    semi-major axis a and flattening f.
    Parameters
    ----------
    lon1 : float, array_like(m,), array_like(m,n)
        longitude of source points in radians
    lat1 : float, array_like(m,), array_like(m,n)
        latitude of source points in radians
    lon2 : float, array_like(m,), array_like(m,n)
        longitude of target points in radians
    lat2 : float, array_like(m,), array_like(m,n)
        latitude of target points in radians
    a : float
        semi-major axis of ellipsoid
    f : float
        flattening of ellipsoid

    Returns
    -------
    d : ndarray(m,), ndarray(m,n)
        ellipsoidal distance between points (lon1, lat1) and (lon2, lat2) in meters

    Notes
    -----
    This function uses the approximation formula by Lambert [1]_ and gives meter level accuracy.

    References
    ----------
    .. [1] Lambert, W. D (1942). "The distance between two widely separated points
           on the surface of the earth". J. Washington Academy of Sciences. 32 (5): 125–130.

    """
    beta1 = np.arctan((1 - f) * np.tan(lat1))
    beta2 = np.arctan((1 - f) * np.tan(lat2))

    sigma = spherical_distance(lon1, beta1, lon2, beta2, r=1)

    P = (beta1 + beta2) * 0.5
    Q = (beta2 - beta1) * 0.5

    X = (sigma - np.sin(sigma)) * ((np.sin(P) * np.cos(Q)) / np.cos(sigma*0.5))**2
    Y = (sigma + np.sin(sigma)) * ((np.cos(P) * np.sin(Q)) / np.sin(sigma*0.5))**2

    return a * (sigma - 0.5 * f * (X + Y))
