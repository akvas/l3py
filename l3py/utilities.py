# Copyright (c) 2018 Andreas Kvas
# See LICENSE for copyright/license details.


import datetime as dt
import calendar as cal


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
    current = dt.datetime(start.year, start.month, round(cal.monthrange(start.year, start.month)[1]*0.5) if use_middle else 1) # set day to 1
    while current < stop:
        yield current
        roll_over = (current.month == 12)

        next_year = current.year+1 if roll_over else current.year
        next_month = 1 if roll_over else current.month+1
        next_day = round(cal.monthrange(next_year, next_month)[1]*0.5) if use_middle else current.day

        current = dt.datetime(next_year, next_month, next_day)
