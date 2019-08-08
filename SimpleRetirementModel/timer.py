import numpy as np
import pandas as pd
import time as tt
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets

def timeUnits(times, unit = "ms", ndigits = 5):
    """
    Function formats times for the bench function
    :param times: list of times to be formatted
    :param unit: string the units of time: "ms": (milliseconds), "us": (microseconds),
                    "ns": nanoseconds, "s": time in seconds, "raw": raw time in seconds
    :param ndigits: number of decimal places to round down the times
    :return: list of formatted times
    """
    if unit == "ms":
        return [round(i * 1E3, ndigits) for i in times]
    elif unit == "us":
        return [round(i * 1E6, ndigits) for i in times]
    elif unit == "ns":
        return [round(i * 1E9, ndigits) for i in times]
    elif unit == "s":
        return [round(i, ndigits) for i in times]
    elif unit == "raw":
        return times

def bench(sExpr, neval=100, units = "ms", ndigits = 5):
    """
    :param expr: string expression to be evaluated
    :param neval: number of evaluations that the statistics will be calculated from
    :param units: string the units of time: "ms": (milliseconds), "us": (microseconds),
                    "ns": nanoseconds, "s": time in seconds, "raw": raw time in seconds
    :param ndigits: number of decimal places to round down the times
    :return: Tuple of times min, lower, mid, upper quartiles and max time, and the expression run
    """
    times = np.ndarray(shape=(neval,), dtype = np.float64)
    expr = compile(sExpr, "<string>", "eval")
    for i in np.arange(neval):
        start = tt.time()
        out = eval(expr)
        end = tt.time()
        times[i] = end - start
    times = np.percentile(times, [0, 25, 50, 75, 100])
    times = timeUnits(times, units, ndigits)

    summ = [sExpr, times[0], times[1], times[2], times[3], times[4], neval]    
    return summ

def lbenchmark(lExpr, plot=False, **kwargs):
    """
    List version of benchmark, takes in a list of string expressions as lExpr

    :param lExpr: list of strings to be evaluated
    :param plot: bool of whether results should be plotted or shown in table
    :param neval: number of evaluations that the statistics will be calculated from
    :param units: string the units of time: "ms": (milliseconds), "us": (microseconds),
                    "ns": nanoseconds, "s": time in seconds, "raw": raw time in seconds
    :param ndigits: number of decimal places to round down the times
    :return: SimpleTable of times min, lower, mid, upper quartiles and max time, and the expression run
    """
    nExpr = lExpr.__len__()
    header = ['expr', 'min', 'lq', 'median', 'uq', 'max', 'neval']
    
    if plot == True:
        out = pd.DataFrame(columns=header)
        for i in np.arange(0,nExpr):
            out.loc[i] = bench(lExpr[i], **kwargs)

    else:
        out = PrettyTable(header)
        for i in np.arange(0,nExpr):
            out.add_row(bench(lExpr[i], **kwargs))

    print(out)

def x_iter(lx, times):
    j = 0
    for i in lx:
        if j == 0:
            out = [len(i)] * times
            j = 1
            
        else:
            out = out + [len(i)] * times
        
    return out    

def bench_plot(lExpr, lx, **kwargs):
    for i in np.arange(0, len(lx)):

        x = lx[i]
        if i == 0:
            df = lbenchmark(lExpr, plot=True, **kwargs)
        
        else:
            df = df.append(lbenchmark(lExpr, plot=True, **kwargs), ignore_index=True)
        
    df['points'] = x_iter(lx, len(lExpr))
    df.pivot_table(index='points', columns='expr', values='median').plot()
