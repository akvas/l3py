# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

"""
Representations of Earth's gravity field.
"""

import numpy as np
import l3py.grid
import l3py.kernel
from l3py.utilities import legendre_functions


def degree_indices(n):
    """
    Return array indices for all coeffcients of degree n.

    The coefficients are ordered by trigonometric function (first all cosine coefficients then all sine coefficients)
    with increasing order.

    Parameters
    ----------
    n : int
        degree

    Returns
    -------
    rows : array_like(k,)
        row indices of all coefficients of degree n
    columns : array_like(k,)
        column indices of all coefficients of degree n

    """

    rows = np.concatenate((np.full(n + 1, n, dtype=int), np.arange(n, dtype=int)))
    columns = np.concatenate((np.arange(n + 1, dtype=int), np.full(n, n, dtype=int)))

    return rows, columns


def order_indices(maximum_degree, m):
    """
    Return array indices for all coeffcients of order m.

    The coefficients are ordered by trigonometric function (first all cosine coefficients then all sine coefficients)
    with increasing degree.

    Parameters
    ----------
    maximum_degree : int
        maximum degree of target coefficient array

    m : int
        order

    Returns
    -------
    rows : array_like(k,)
        row indices of all coefficients of order m
    columns : array_like(k,)
        column indices of all coefficients of order m

    """
    rows = np.arange(m, maximum_degree + 1, dtype=int)
    columns = np.full(rows.size, m)

    if m > 0:
        rows = np.concatenate((rows, np.full(maximum_degree + 1 - m, m - 1)))
        columns = np.concatenate((columns, np.arange(m, maximum_degree + 1, dtype=int)))

    return rows, columns


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


class PotentialCoefficients:
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
        """Return a deep copy of the PotentialCoefficients instance."""
        gf = PotentialCoefficients(self.GM, self.R)
        gf.anm = self.anm.copy()
        gf.epoch = self.epoch

        return gf

    def slice(self, min_degree=None, max_degree=None, min_order=None, max_order=None, step_degree=1, step_order=1):
        """
        Slice a PotentialCoefficients instance to a specific degree and order range. Return value is a new
        PotentialCoefficients instance, the original gravity field is unchanged.

        Parameters
        ----------
        min_degree : int
            minimum degree of sliced PotentialCoefficients (Default: 0)
        max_degree : int
            maximum degree of sliced PotentialCoefficients (Default: maximum degree if calling object)
        min_order : int
            minimum order of sliced PotentialCoefficients (Default: 0)
        max_order : int
            maximum order of sliced PotentialCoefficients (Default: max_degree)
        step_degree : int
            step between min_degree and max_degree (Default: 1)
        step_order : int
            step between min_order and max_order (Default: 1)

        Returns
        -------
        gravityfield : PotentialCoefficients
            new PotentialCoefficients instance with all coefficients outside of the passed degree and order ranges
            set to zero

        """

        min_degree = 0 if min_degree is None else min_degree
        max_degree = self.nmax() if max_degree is None else max_degree
        min_order = 0 if min_order is None else min_order
        max_order = max_degree if max_order is None else max_order

        idx_degree = np.isin(self.__degree_array(), range(min_degree, max_degree + 1, step_degree))
        idx_order = np.isin(self.__order_array(), range(min_order, max_order + 1, step_order))

        gf = PotentialCoefficients(self.GM, self.R)
        gf.epoch = self.epoch
        gf.anm = np.zeros(self.anm.shape)
        gf.anm[np.logical_and(idx_degree, idx_order)] = self.anm[np.logical_and(idx_degree, idx_order)].copy()

        gf.truncate(max_degree)

        return gf

    def append(self, *coeffs):
        """Append a coefficient to a PotentialCoefficients instance."""
        for coeff in coeffs:
            if coeff.n > self.nmax():
                tmp = np.zeros((coeff.n+1, coeff.n+1))
                tmp[0:self.anm.shape[0], 0:self.anm.shape[1]] = self.anm.copy()
                self.anm = tmp

            if coeff.trigonometric_function == np.cos:
                self.anm[coeff.n, coeff.m] = coeff.value
            elif coeff.trigonometric_function == np.sin and coeff.m > 0:
                self.anm[coeff.m-1, coeff.n] = coeff.value

    def truncate(self, nmax):
        """Truncate a PotentialCoefficients instance to a new maximum spherical harmonic degree."""
        if nmax < self.nmax():
            self.anm = self.anm[0:nmax+1, 0:nmax+1]

    def replace_c20(self, gravityfield):
        """
        Replace the c20 coefficient of a PotentialCoefficients instance by c20 of another
        PotentialCoefficients instance. Substitution is performed in-place.

        Parameters
        ----------
        gravityfield : PotentialCoefficients instance
            gravity field containing the new c20 value

        """
        self.anm[2, 0] = gravityfield.anm[2, 0]

    def coefficients_by_degree(self, n):
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

    def __degree_array(self):
        """Return degrees of all coefficients as numpy array"""
        da = np.zeros(self.anm.shape, dtype=int)
        for n in range(self.nmax()+1):
            da[n, 0:n+1] = n
            da[0:n, n] = n

        return da

    def __order_array(self):
        """Return orders of all coefficients as numpy array"""
        da = np.zeros(self.anm.shape, dtype=int)
        for m in range(1, self.nmax()+1):
            da[m - 1, m::] = m
            da[m::, m] = m

        return da

    def nmax(self):
        """Return maximum spherical harmonic degree of a PotentialCoefficients instance."""
        return self.anm.shape[0]-1

    def __add__(self, other):
        """Coefficient-wise addition of two PotentialCoefficients instances."""
        if not isinstance(other, PotentialCoefficients):
            raise TypeError("unsupported operand type(s) for +: '"+str(type(self))+"' and '"+str(type(other))+"'")

        factor = (other.R / self.R) ** other.__degree_array() * (other.GM / self.GM)
        if self.nmax() >= other.nmax():
            result = self.copy()
            result.anm[0:other.anm.shape[0], 0:other.anm.shape[1]] += (other.anm*factor)
        else:
            result = PotentialCoefficients(self.GM, self.R)
            result.epoch = self.epoch
            result.anm = other.anm*factor
            result.anm[0:self.anm.shape[0], 0:self.anm.shape[1]] += self.anm

        return result

    def __sub__(self, other):
        """Coefficient-wise subtraction of two PotentialCoefficients instances."""
        if not isinstance(other, PotentialCoefficients):
            raise TypeError("unsupported operand type(s) for -: '"+str(type(self))+"' and '"+str(type(other))+"'")

        return self+(other*-1)

    def __mul__(self, other):
        """Multiplication of a PotentialCoefficients instance with a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for *: '"+str(type(self))+"' and '"+str(type(other))+"'")

        result = self.copy()
        result.anm *= other

        return result

    def __truediv__(self, other):
        """Division of a PotentialCoefficients instance by a numeric scalar."""
        if not isinstance(other, (int, float)):
            raise TypeError("unsupported operand type(s) for /: '"+str(type(self))+"' and '"+str(type(other))+"'")

        return self*(1.0/other)

    def degree_amplitudes(self, kernel='potential'):
        """
        Compute degree amplitudes from potential coefficients.

        Parameters
        ----------
        kernel : string
            name of kernel for the degree amplitude computation

        Returns
        -------
        degrees : array_like shape (self.nmax()+1,)
            integer sequence of degrees
        amplitudes : array_like shape (self.nmax()+1,)
            computed degree amplitudes
        """
        degrees = np.arange(self.nmax()+1)
        amplitudes = np.zeros(degrees.size)

        kn = l3py.kernel.get_kernel(kernel, self.nmax())

        for n in degrees:
            amplitudes[n] = np.sum(self.anm[l3py.gravityfield.degree_indices(n)]**2)*kn.kn(n)**2

        return degrees, np.sqrt(amplitudes)*self.GM/self.R

    def coefficient_triangle(self, min_degree=2, max_degree=None):
        """
        Arrange spherical harmonic coefficients as triangle for visualization.

        Parameters
        ----------
        min_degree : int
            degrees below min_degree are masked out
        max_degree : int
            triangle is truncated at max_degree

        Returns
        -------
        triangle : masked_array shape (max_degree+1, 2*max_degree-1)

        """

        max_degree = self.nmax() if max_degree is None else max_degree

        triangle = np.hstack((np.rot90(self.anm, -1), self.anm))
        mask = np.hstack((np.rot90(np.tril(np.ones(self.anm.shape, dtype=bool)), -1),
                               np.triu(np.ones(self.anm.shape, dtype=bool), 1)))
        mask[0:min_degree] = True

        return np.ma.masked_array(triangle, mask=mask)[0:max_degree+1, :]

    def to_grid(self, grid=l3py.grid.GeographicGrid(), kernel='ewh'):
        """
        Compute gridded values from a set of potential coefficients.

        Parameters
        ----------
        gravityfield : PotentialCoefficients instance
            potential coefficients to be gridded
        grid : instance of Grid subclass
            point distribution (Default: 0.5x0.5 degree geographic grid)
        kernel : string
            gravity field functional to be gridded (Default: equivalent water height). See Kernel for details.

        Returns
        -------
        output_grid : instance of type(grid)
            deep copy of the input grid with the gridded values
        """
        inverse_coefficients = l3py.kernel.get_kernel(kernel, self.nmax())

        if grid.is_regular():
            P = legendre_functions(self.nmax(), grid.colatitude())
            Rr = (self.R / grid.radius())

            gridded_values = np.zeros((grid.lats.size, grid.lons.size))

            for n in range(self.nmax() + 1):
                coeffs = self.coefficients_by_degree(n)
                orders = [c.m for c in coeffs]
                idx = [int(n * (n + 1) * 0.5 + m) for m in orders]

                kn = inverse_coefficients.kn(n, grid.radius(), grid.colatitude())

                CS = np.vstack([c.trigonometric_function(c.m * grid.lons) * c.value for c in coeffs])
                gridded_values += (kn * Rr ** (n + 1))[:, np.newaxis] * P[:, idx] @ CS

            output_grid = grid.copy()
            output_grid.values = gridded_values*(self.GM/self.R)
            output_grid.epoch = self.epoch
        else:
            raise NotImplementedError('Propagation to arbitrary point distributions is not yet implemented.')

        return output_grid





