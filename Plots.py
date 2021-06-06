import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import Machine_learning as ml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# All plotting functions are collected in this file as well as everything else (mat)plot related

# Courtesy of Wouter Dobbels
def estimate_density(x, y, method='scott', sortPoints=True, subSample=None,
                     verbose=False, logspace=False):
    """
    Useful for scatterplots. This makes it possible to plot high density 
    regions with another color instead of just having overlapping dots.

    Parameters
    ----------
    x, y : array-like
        These are combined into a 2D vector to estimate the density.
    method : str
        Guassian kde estimation technique. 
    sortPoints : bool
        Reorders x and y so high density points come on top. 
    subSample : int or None
        Take a subsample of the points, in order to avoid
        long calculations. Above 10 000 points, the calculation takes some 
        time. 
    logspace : bool, default False
        If True, x and y are transformed to log space before calculating the density.

    Returns
    -------
    x, y, density : array-like
        These match index by index. x and y are returned because they can be 
        reordered or resampled.
    """

    # Convert pandas series to numpy array
    if isinstance(x, pd.core.series.Series):
        x = x.values
    if isinstance(y, pd.core.series.Series):
        y = y.values
    assert x.shape == y.shape, "x and y must have equal lengths"
    if logspace:
        x, y = np.log(x), np.log(y)
    if subSample is not None:
        idx = np.random.choice(x.shape[0], size=subSample, replace=False)
        x, y = x[idx], y[idx]
    if verbose:
        print('Calculating density...')
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=method)
    if verbose:
        print('Kernel factor = {}'.format(kde.factor))
    z = kde(xy)
    if logspace:
        x, y = np.exp(x), np.exp(y)

    # Sort the points by density, so that the densest points are plotted last
    if not sortPoints:
        return z
    if verbose:
        print('Sorting by density')
    idx = z.argsort()
    if isinstance(x, pd.Series):
        return x.iloc[idx], y.iloc[idx], z[idx]
    return x[idx], y[idx], z[idx]


def set_defaults(Default=False, cm='inferno'):
    """
    Function to change some plot parameters and make the plot look nicer.
    This could have easily been done with a .mplstyle file as well.
    fontsize and stuff is made to look pretty on a 4k screen. adjust as needed

    Parameters
    ----------
    Default: bool
        if True, sets everything to the default matplotlib style
    cm: string, default='inferno'
        must be the name of a matplotlib colormap. Is then used as the colormap
        and creates a cycler based on the colormap
        https://matplotlib.org/stable/tutorials/colors/colormaps.html for a list of all colormaps

    returns
    -------
    a list of colors, usable for plots where the prop_cycle doesn't work
    """
    # set back to default value
    if Default:
        mpl.style.use('default')

    # Inferno has been used to base all colors on
    else:
        cmap_str = cm
        mpl.rc('image', cmap=cmap_str)
        cmap = mpl.cm.get_cmap(cmap_str)
        mpl.rc('axes', axisbelow=True, labelsize="x-large", titlesize='xx-large')
        mpl.rc('axes', prop_cycle=cycler('color', [cmap(i) for i in np.arange(0.15, 1.05, 0.15)]))
        mpl.rc('grid', lw=0.5, c='k', alpha=0.5)
        mpl.rc('savefig', dpi=320)
        mpl.rc('font', size=12)

    return [cmap(i) for i in np.arange(0.15, 1.05, 0.15)]


def scatter_with_hist(x, y, figure=1, density=True, grid=True):
    """
    Makes a scatterplot with the corresponding histograms on the left and top

    Parameters
    ----------
    x, y: array-like
        the points which are used to make the plots
    figure: int
        controls the plt.figure parameter
    density: bool
        If true, uses estimate_density to create a density scatterplot
        usefull for highly overlapping data
    grid: bool
        enable or disable the grid
    
    Returns
    -------
    ax_scatter, ax_histx, ax_histy: matplotlib axes objects
        can be used to add titles, labels,...
    """

    if density:
        x, y, c = estimate_density(x, y)

    # definitions for the axes
    left, width = 0.11, 0.65
    bottom, height = 0.095, 0.635
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8), num=figure)

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    if density:
        ax_scatter.scatter(x, y, c=c, alpha=0.4)
    else:
        ax_scatter.scatter(x, y)

    # the histograms
    bins = 15
    ax_histx.hist(x, bins=bins, rwidth=0.95)
    ax_histy.hist(y, bins=bins, orientation='horizontal', rwidth=0.95)

    # limits
    xlim = ax_scatter.get_xlim()
    ylim = ax_scatter.get_ylim()
    lims = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_histx.set_xlim(lims)
    ax_histy.set_ylim(lims)

    # Grids
    if grid:
        ax_scatter.grid()
        ax_histx.grid()
        ax_histy.grid()

    return (ax_scatter, ax_histx, ax_histy)


def CheckMissingData(dataframe, columns, band_names=[], show=True, labels=True, verbose=False):
    """
    creates a histogram denoting how many datapoints in each band are missing

    Parameters
    -----
    dataframe: Pandas Dataframe
        dataframe to check
    columns: (numpy) array
        Bands to check (as written in the dataframe)
    band_names: (numpy) array
        Band names to print on the plot. If empty, uses the same names as the columns of the dataframe
    show: bool
        if true, immediately plots the created plot. else use plt.plot() after the function to plot
    labels: bool
        automatically generate labels if true
    verbose: bool
        when true, prints what the program is currently doing
    
    Returns
    -------
    No returns, only a histogram is created using matplotlib
    """
    
    if band_names == []:
        band_names = columns

    print("Checking missing data...") if verbose else None

    zero_list = np.zeros(len(columns))
    all_zero = np.zeros(len(columns))

    # Determine which bands have missing data by looking if they have zero flux measured.
    # Total number of missing data in each band is then counted
    for i, row in dataframe.iterrows():
        zeros = np.isclose(row[columns].tolist(), all_zero)
        zero_list = zero_list + zeros

    print("Plotting...") if verbose else None

    plt.bar(np.linspace(0, 21, num=21, endpoint=False),zero_list, width=0.95)
    if labels:
        plt.xticks(np.linspace(0, 21, num=21, endpoint=False), labels=band_names, rotation=45, fontsize='small')
        plt.ylabel("Number of galaxies")
        plt.title("missing data per band")
        bottom, top = plt.ylim()
        #plt.text(-0.5, 0.9*top, 'Total data: {:g} galaxies'.format(len(dataframe.index)), fontsize='medium')
    if show:
        plt.show()
    return None


def add_subplot_axes(ax,rect,facecolor='w'):
    """
    Function to add plots-in-plots to subplots, courtesy of StackOverflow user Pablo
    Modified to make it compatible with matplotlib 2.0+ (but no longer with older versions I think)
    See the discussion for more info
    https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    """

    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax