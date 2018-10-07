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
    print(grace_monthly.strftime('Compute filtered grid for %Y-%m.'))
    filtered_solution = shc_filter.filter(grace_monthly)
    filtered_grid = l3py.grid.shc2grid(filtered_solution)
    grids.append(filtered_grid)

l3py.io.savegrids('output.nc', grids)

