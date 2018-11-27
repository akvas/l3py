# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.

"""
File I/O for gravity field representations and gridded data.
"""

from l3py.gravityfield import PotentialCoefficients, Coefficient
from netCDF4 import Dataset
import datetime as dt
import numpy as np


def __parse_gfc_entry(line):
    """Return the values for both coefficients in a GFC file line."""
    sline = line.split()
    n = int(sline[1])
    m = int(sline[2])

    return Coefficient(np.cos, n, m, float(sline[3])), Coefficient(np.sin, n, m, float(sline[4]))


def loadgfc(fname, nmax=None):
    """
    Read a set of potential coefficients from a GFC file.

    Parameters
    ----------
    fname : str
        name of GFC file
    nmax : int
        truncate PotentialCoefficients instance at degree nmax (Default: return the full field)

    Returns
    -------
    gf : PotentialCoefficients
        PotentialCoefficients instance
    """
    gf = PotentialCoefficients()

    with open(fname, 'r') as f:

        for line in f:
            if line.startswith('gfc'):
                cnm, snm = __parse_gfc_entry(line)
                gf.append(cnm, snm)

            elif line.startswith('radius'):
                gf.R = float(line.split()[-1])
            elif line.startswith('earth_gravity_constant'):
                gf.GM = float(line.split()[-1])

    if nmax is not None:
        gf.truncate(nmax)

    return gf


def savegrids(fname, grids, reference_epoch=None):
    """
    Write a list of Grid instances to a netCDF file. In order to properly define the time coordinate, make sure
    the `epoch` member of each grid in the passed container is set.

    Parameters
    ----------
    fname : str
        name of netCDF file
    grids : list of Grid instances
        list of grids to be written to file
    reference_epoch : datetime
        reference epoch for time axis (Default: use January 1st of year of first grid)
    """

    if len(grids) == 0:
        raise RuntimeWarning('Empty container of grids passed.')

    dataset = Dataset(fname, 'w')

    dataset.createDimension('lon', grids[0].lons.size)
    dataset.createDimension('lat', grids[0].lats.size)
    dataset.createDimension('time', None)

    ref_grid = grids[0]
    lats = dataset.createVariable('lat', float, ('lat',))
    lats.standard_name = 'latitude'
    lats.long_name = 'latitude'
    lats.units = 'degrees_north'
    lats.axis = 'Y'
    lats[:] = ref_grid.lats*180/np.pi

    lons = dataset.createVariable('lon', float, ('lon',))
    lons.standard_name = 'longitude'
    lons.long_name = 'longitude'
    lons.units = 'degrees_east'
    lons.axis = 'X'
    lons[:] = ref_grid.lons*180/np.pi

    epochs = [grid.epoch for grid in grids]
    if reference_epoch is None:
        reference_epoch = dt.datetime(epochs[0].year, 1, 1)
    times = dataset.createVariable('time', float, ('time',))
    times.standard_name = 'time'
    times.units = reference_epoch.strftime('days since %Y-%m-%d %H:%M:%S')
    times.axis = 'T'
    times[:] = [(t-reference_epoch).days for t in epochs]

    values = dataset.createVariable('ewh', float, ('time', 'lat', 'lon'))
    values.long_name = 'equivalent water height'
    values.units = 'm'
    values[:, :, :] =[grid.values for grid in grids]

    dataset.close()

