# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Auxiliary functions.
"""

import datetime as dt
import calendar as cal
import numpy as np


def month_iterator(start, stop, use_middle=False):
    """
    Generator for a monthly sequence of datetime objects.

    To be consistent with Python ranges, the last epoch generated will be strictly less than `stop`.

    Parameters
    ----------
    start : datetime object
        epoch from which the first month will be generated
    stop : datetime object
        epoch from which the last month will be generated (last month will be strictly less than stop)
    use_middle : bool
        If True, the midpoint of each month will be returned, otherwise the first of each month is used (default: False)

    Returns
    -------
    g : Generator object
        Generator for monthly datetime objects

    """
    current = dt.datetime(start.year, start.month,
                          round(cal.monthrange(start.year, start.month)[1] * 0.5) if use_middle else 1)  # set day to 1
    while current < stop:
        yield current
        roll_over = (current.month == 12)

        next_year = current.year + 1 if roll_over else current.year
        next_month = 1 if roll_over else current.month + 1
        next_day = round(cal.monthrange(next_year, next_month)[1] * 0.5) if use_middle else current.day

        current = dt.datetime(next_year, next_month, next_day)


def legendre_functions(nmax, colat):
    """
    Associated fully normalized Legendre functions (1st kind).

    Parameters
    ----------
    nmax : int
        maximum spherical harmonic degree to compute
    colat : float, array_like(m,)
        co-latitude of evaluation points in radians

    Returns
    -------
    Pnm : array_like(m, nmax + 1, nmax + 1)
        Array containing the fully normalized Legendre functions. Pnm[:, n, m] returns the
        Legendre function of degree n and order m for all points, as does Pnm[:, m-1, n] (for m > 0).

    """
    theta = np.atleast_1d(colat)
    function_array = np.zeros((theta.size, nmax + 1, nmax + 1))

    function_array[:, 0, 0] = 1.0  # initial values for recursion
    function_array[:, 1, 0] = np.sqrt(3) * np.cos(theta)
    function_array[:, 1, 1] = np.sqrt(3) * np.sin(theta)

    for n in range(2, nmax + 1):
        function_array[:, n, n] = np.sqrt((2.0 * n + 1.0) / (2.0 * n)) * np.sin(theta) * \
                                  function_array[:, n - 1, n - 1]

    index = np.arange(nmax + 1)
    function_array[:, index[2:], index[1:-1]] = np.sqrt(2 * index[2:] + 1) * np.cos(theta[:, np.newaxis]) * \
                                                function_array[:, index[1:-1], index[1:-1]]

    for row in range(2, nmax + 1):
        n = index[row:]
        m = index[0:-row]
        function_array[:, n, m] = np.sqrt((2.0 * n - 1.0) / (n - m) * (2.0 * n + 1.0) / (n + m)) * \
                                  np.cos(theta[:, np.newaxis]) * function_array[:, n - 1, m] - \
                                  np.sqrt((2.0 * n + 1.0) / (2.0 * n - 3.0) * (n - m - 1.0) / (n - m) *
                                          (n + m - 1.0) / (n + m)) * function_array[:, n - 2, m]

    for m in range(1, nmax + 1):
        function_array[:, m - 1, m:] = function_array[:, m:, m]

    return function_array


def normal_gravity(r, colat, a=6378137.0, f=298.2572221010 ** -1, convergence_threshold=1e-9):
    """
    Normal gravity on the ellipsoid (GRS80).

    Parameters
    ----------
    r : float, array_like, shape(m, )
        radius of evaluation point(s) in meters
    colat : float, array_like, shape (m,)
        co-latitude of evaluation points in radians
    a : float
        semi-major axis of ellipsoid (Default: GRS80)
    f : float
        flattening of ellipsoid (Default: GRS80)
    convergence_threshold : float
        maximum absolute difference between latitude iterations in radians

    Returns
    -------
    g : float, array_like, shape(m,) (depending on types of r and colat)
        normal gravity at evaluation point(s) in [m/s**2]
    """
    ga = 9.7803267715
    gb = 9.8321863685
    m = 0.00344978600308

    z = np.cos(colat) * r
    p = np.abs(np.sin(colat) * r)

    b = a * (1 - f)
    e2 = (a / b - 1) * (a / b + 1)
    latitude = np.arctan2(z * (1 + e2), p)

    L = np.abs(latitude) < 60 / 180 * np.pi

    latitude_old = np.full(latitude.shape, np.inf)
    h = np.zeros(latitude.shape)

    while np.max(np.abs(latitude - latitude_old)) > convergence_threshold:
        latitude_old = latitude.copy()

        N = (a / b) * a / np.sqrt(1 + e2 * np.cos(latitude) ** 2)
        h[L] = p[L] / np.cos(latitude[L]) - N[L]
        h[~L] = z[~L] / np.sin(latitude[~L]) - N[~L] / (1 + e2)

        latitude = np.arctan2(z * (1 + e2), p * (1 + e2 * h / (N + h)))

    cos2 = np.cos(latitude) ** 2
    sin2 = np.sin(latitude) ** 2

    gamma0 = (a * ga * cos2 + b * gb * sin2) / np.sqrt(a ** 2 * cos2 + b ** 2 * sin2)
    return gamma0 - 2 * ga / a * (1 + f + m + (-3 * f + 5 * m / 2) * sin2) * h + 3 * ga / a ** 2 * h ** 2


def geocentric_radius(latitude, a=6378137.0, f=298.2572221010 ** -1):
    """
    Geocentric radius of a point on the ellipsoid.

    Parameters
    ----------
    latitude : float, array_like, shape(m, )
       latitude of evaluation point(s) in radians
    a : float
       semi-major axis of ellipsoid (Default: GRS80)
    f : float
       flattening of ellipsoid (Default: GRS80)

    Returns
    -------
    r : float, array_like, shape(m,) (depending on type latitude)
       geocentric radius of evaluation point(s) in [m]
    """
    e2 = 2 * f * (1 - f)
    nu = a / np.sqrt(1 - e2 * np.sin(latitude) ** 2)

    return nu * np.sqrt(np.cos(latitude) ** 2 + (1 - e2) ** 2 * np.sin(latitude) ** 2)


def colatitude(latitude, a=6378137.0, f=298.2572221010 ** -1):
    """
    Co-latitude of a point on the ellipsoid.

    Parameters
    ----------
    latitude : float, array_like, shape(m, )
      latitude of evaluation point(s) in radians
    a : float
      semi-major axis of ellipsoid (Default: GRS80)
    f : float
      flattening of ellipsoid (Default: GRS80)

    Returns
    -------
    psi : float, array_like, shape(m,) (depending on type latitude)
      colatitude of evaluation point(s) in [rad]
    """
    e2 = 2 * f * (1 - f)
    nu = a / np.sqrt(1 - e2 * np.sin(latitude) ** 2)

    return np.arccos(nu * (1 - e2) * np.sin(latitude) / geocentric_radius(latitude, a, f))
