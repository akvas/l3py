"""
Example: Compute DDK3 filtered grids for ocean applications

The test data required is available on the ITSG FTP server:
ftp://ftp.tugraz.at/outgoing/ITSG/GRACE/ITSG-Grace2018/monthly

Download the potential coefficients to a directory of your choice and change the variables `model_dir` and
`solution_dir` accordingly.

The example script performs the following steps for a time series of GRACE monthly solutions:

- replace c20 with an SLR estimate
- add degree one coefficients
- remove GIA signal
- restored AOD1B ocean bottom pressure to get the full ocean mass

- reduce the mean of the resulting time series
- compute gridded equivalent water height on a 0.5x0.5 degree geographic grid
- write the grid time series to a netCDF file

"""

import l3py
import numpy as np
import datetime as dt


time_start = dt.datetime(2002, 1, 1)
time_end = dt.datetime(2017, 1, 1)

epochs = [t for t in l3py.utilities.month_iterator(time_start, time_end, use_middle=True)]

model_dir = 'monthly_background'
solution_dir = 'monthly_n96'

monthly_solutions = []
for t in epochs:

    try:
        grace_monthly = l3py.io.loadgfc(t.strftime(solution_dir + '/ITSG-Grace2018_n96_%Y-%m.gfc'))

        slr_c20 = l3py.io.loadgfc(t.strftime(model_dir + '/model_c20_%Y-%m.gfc'))
        degree1 = l3py.io.loadgfc(t.strftime(model_dir + '/model_degree1_%Y-%m.gfc'))
        gia = l3py.io.loadgfc(t.strftime(model_dir + '/model_glacialIsostaticAdjustment_%Y-%m.gfc'))
        obp = l3py.io.loadgfc(t.strftime(model_dir + '/model_oceanBottomPressure_%Y-%m.gfc'))

        gia.truncate(grace_monthly.nmax())
        obp.truncate(grace_monthly.nmax())

        grace_monthly.replace_c20(slr_c20)
        grace_monthly += degree1

        grace_monthly -= gia
        grace_monthly += obp

        grace_monthly.epoch = t
        monthly_solutions.append(grace_monthly)

    except FileNotFoundError as e:
        print(e)
        continue

monthly_solutions = np.array(monthly_solutions)
monthly_solutions -= np.mean(monthly_solutions)

shc_filter = l3py.filter.DDK(3)

base_grid = l3py.grid.GeographicGrid(dlon=0.5, dlat=0.5)

grids = []
for grace_monthly in monthly_solutions:
    print(grace_monthly.epoch.strftime('Compute filtered grid for %Y-%m.'))

    filtered_solution = shc_filter.filter(grace_monthly)
    filtered_grid = filtered_solution.to_grid(base_grid, kernel='ewh')
    grids.append(filtered_grid)

l3py.io.savegrids('output.nc', grids)

