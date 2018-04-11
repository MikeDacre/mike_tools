"""
A collection of plotting functions to use with pandas, numpy, and pyplot.

       Created: 2016-36-28 11:10
"""
import sys
from operator import itemgetter
from itertools import groupby, cycle

import numpy as np
import scipy as sp
import scipy.stats as sts
import pandas as pd

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import matplotlib as mpl
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

import networkx as nx
from adjustText import adjust_text

import matplotlib_venn

# Get rid of pandas future warnings
import warnings as _warn
_warn.simplefilter(action='ignore', category=FutureWarning)
_warn.simplefilter(action='ignore', category=UserWarning)
_warn.simplefilter(action='ignore', category=RuntimeWarning)


###############################################################################
#                                 Regression                                  #
###############################################################################


class LinearRegression(object):

    """
    Use statsmodels, scipy, and numpy to do a linear regression.

    Regression includes confidence intervals and prediction intervals.

    Attributes
    ----------
    {x,y}
        Original x, y data.
    X : pandas.core.frame.DataFrame
        Original X with a column of ones added as a constant (unless x already
        had a constant column).
    x_pred : numpy.ndarray
        Predicted evently spaced data. From
        ``numpy.linspace(x.min(), x.max(), len(x))``
    x_pred_c : numpy.ndarry
        Original x_pred with a column of ones added as a constant
    y_pred :numpy.ndarray
        Predicted y values from the model
    model : statsmodels.regression.linear_model.OLS
    fitted : statsmodels.regression.linear_model.RegressionResultsWrapper
        Result of model.fit()
    summary : statsmodels.iolib.summary.Summary
        Summary data for regression
    reg_line : numpy.ndarray
        Statsmodels fitted values as an ndarray for plotting
    p : float
        P value for regression in x
    rsquared : float
        The r-squared for the regression
    conf : numpy.ndarray
        Confidence interval array for x values.
    ci_{upper,lower} : numpy.ndarray
        Upper and lower confidence interval arrays. 95% chance real regression
        line in this interval
    ci_t : float
        T statistic used for the confidence interval. 95% chance real y value
        is in this interval
    pred_{upper,lower} : numpy.ndarray
        Upper and lower prediction
    {y_hat,y_err,s_err,sdev} : numpy.ndarray
    """

    def __init__(self, x, y):
        """
        Do the regression.

        Params
        ------
        {x,y} : numpy.ndarray or pandas.core.series.Series
            X and Y data
        log : bool
            Do the regression in log10 space instead
        """
        self.x = x
        self.y = y
        self.n = len(x)
        self.X = sm.add_constant(x)

        # Do regression
        self.model = sm.OLS(self.y, self.X)
        self.fitted = self.model.fit()
        self.reg_line = self.fitted.fittedvalues
        self.reg_summary = self.fitted.summary()

        # Get predicted data
        self.x_pred = np.linspace(x.min(), x.max(), self.n)
        self.x_pred_c = sm.add_constant(self.x_pred)
        self.y_pred = self.fitted.predict(self.x_pred_c)

        # Calculate confidence intervals
        self.y_hat = self.fitted.predict(self.X)
        self.y_err = y - self.y_hat
        mean_x = self.x.T[1].mean()
        dof = self.n - self.fitted.df_model - 1

        self.ci_t = sts.t.ppf(1-0.025, df=dof)
        self.s_err = np.sum(np.power(self.y_err, 2))

        self.conf = (
            self.ci_t * np.sqrt(
                (
                    self.s_err/(self.n-2)
                )*(
                    1.0/self.n + (
                        np.power(
                            (self.x_pred-mean_x), 2
                        )/(
                            (np.sum(np.power(self.x_pred,2))) - self.n*(np.power(mean_x,2))
                        )
                    )
                )
            )
        )
        self.ci_upper = self.y_pred + abs(self.conf)
        self.ci_lower = self.y_pred - abs(self.conf)

        # Get prediction intervals
        self.sdev, self.pred_lower, self.pred_upper = wls_prediction_std(
            self.fitted, exog=self.x_pred_c, alpha=0.05
        )

        # Assign stats
        self.rsquared = self.fitted.rsquared
        self.P = self.fitted.pvalues.tolist()[0]

    def plot_reg_line(self, ax, alpha=0.7, zorder=12, color=None,
                      include_label=True, unlog=False):
        """Plot the regression line."""
        color = color if color else 'darkorchid'
        x_pred = 10**self.x_pred if unlog else self.x_pred
        y_pred = 10**self.y_pred if unlog else self.y_pred
        label = self.legend_text() if include_label else None
        ax.plot(
            x_pred, y_pred, '-', color=color, linewidth=2,
            label=label, alpha=alpha, zorder=zorder
        )

    def plot_ci_line(self, ax, alpha=0.3, zorder=10, color=None, unlog=False):
        """Plot the confidence interval lines."""
        color = color if color else sns.xkcd_rgb['rust']
        x_pred = 10**self.x_pred if unlog else self.x_pred
        ci_upper = 10**self.ci_upper if unlog else self.ci_upper
        ci_lower = 10**self.ci_lower if unlog else self.ci_lower
        ax.fill_between(
            x_pred, ci_lower, ci_upper, color=color,
            alpha=alpha, zorder=zorder
        )

    def plot_pred_line(self, ax, alpha=0.1, zorder=5, color=None, unlog=False):
        """Plot the confidence interval lines."""
        color = color if color else sns.xkcd_rgb['light green']
        x_pred = 10**self.x_pred if unlog else self.x_pred
        pred_upper = 10**self.pred_upper if unlog else self.pred_upper
        pred_lower = 10**self.pred_lower if unlog else self.pred_lower
        ax.fill_between(
            x_pred, pred_lower, pred_upper, color=color,
            interpolate=True, alpha=alpha, zorder=zorder
        )

    def print_reg_summary(self):
        """Print the regression summary."""
        print(self.fitted.summary())

    def legend_text(self):
        """Print summary stats."""
        return 'OLS: {:.3} +/- {:.2}\n    P: {:.2e}\n  $R^2$: {:.2}'.format(
            self.fitted.params.tolist()[0], self.fitted.bse.tolist()[0],
            self.P, self.rsquared
        )

    def __str__(self):
        """Return summary stats."""
        return 'OLS: {:.3} +/- {:.2}\nP: {:.2e}\nR-squared: {:.2}'.format(
            self.fitted.params.tolist()[0], self.fitted.bse.tolist()[0],
            self.P, self.rsquared
        )

    def __repr__(self):
        """Print repr for statsmodels."""
        return 'LinearRegression({0}, R2: {1:.2e}, P: {2:.2e})'.format(
            repr(self.model), self.rsquared, self.P
        )


###############################################################################
#                             Basic Scatter Plots                             #
###############################################################################


def distcomp(y, x=None, bins=500, kind='qq', style=None, ylabel=None,
             xlabel=None, title=None, fig=None, ax=None, size=10):
    """Compare two vectors of different length by creating equal bins.

    If kind is qq, the plot is a simple Quantile-Quantile plot, if it is pp,
    then the plot is a cumulative probability plot.

    There are three different formats:
        simple is a single scatter plot of either the QQ or cumulative dist
        joint includes the simple scatter plot, but also plots the contributing
        univatiate distributions on the axes
        column includes the scatter plot on the left and adds two independent
        histograms in two plots in a second column on the right.

    We also add the Mann-Whitney U P-value for the *original distributions*,
    pre-binning.

    Cumulative probability is calculated like this:

        Uses the min(x,y) and max(x,y) (+/- 1%) to set the limit of the bins,
        and then divides both x and y into an equal number of bins between
        those two limits, ensuring that the bin starts and ends are identical
        for both distributions. Bins are labelled by the center value of the
        bin.

        Bins are then converted into frequencies, such that the sum of all
        frequencies is 1. These frequencies are then converted to a cumulative
        frequency, so that the final bin will always have a value of one.


    Parameters
    ----------
    y (actual|y-axis) : Series
        Series of actual data, will go on y-axis
    x (theoretical|x-axis) : Series or {'normal', 'uniform', 'pvalue'}, optional
        Series of theoretical data, will go on x-axis, can also be one of
        'normal', 'uniform', or 'pvalue', to use a random distribution of the
        same length as the y data.
        normal: will use a normal distribution anchored at the mean of y,
        with a scope of the standard deviation of y.
        uniform: will use a uniform distribution min(y) >= dist <= max(y)
        pvalue: will use a uniform distribution between 0 and 1
        Defaults to 'pvalue' if kind='qq_log' else 'normal'
    bins : int, optional
        Number of bins to use for plotting
    kind : {'qq', 'qq_log', 'pp', 'cum', 'lin_pp'}, optional
        qq:  Plot a Q-Q plot
        qq_log:  Plot a Q-Q plot of -log10(pvalues)
        pp|cum: Plot a cumulative probability plot
        lin_pp: Plot a probability plot where bins are evenly spaced
    style : str, optional
        simple: Plot a simple scatter plot
        joint: Plot a scatter plot with univariate dists on each axis.
        column: Plot a scatter plot with univariate histograms, separately
        calculated, on the side.
    {y/x}label : str, optional
        Optional label for y/x axis. y=Actual, x=Theretical
    title : str, optional
        Optional title for the whole plot
    size : int, optional
        A size to use for the figure, square is forced.

    Returns
    -------
    fig, ax
        Figure and axes always returned, if joint is True, axes
        object will be a seaborn axgrid.
    """
    if kind not in ['qq', 'qq_log', 'cum', 'pp', 'lin_pp']:
        raise ValueError('kind must be one of qq, qq_log, pp, cum, or lin_pp')

    if not style:
        if kind is 'qq' or kind is 'qq_log':
            style = 'simple'
        else:
            style = 'joint'

    if x is None:
        if kind is 'qq_log':
            x = 'pvalue'
        else:
            x = 'normal'

    if style not in ['simple', 'joint', 'column', 'scatter']:
        raise ValueError('style must be one of simple, joint, or colummn')

    # Convert to old names
    theoretical = x
    actual = y

    kind = 'cum' if kind == 'pp' else kind
    style = 'simple' if style == 'scatter' else style

    if isinstance(theoretical, str):
        if theoretical == 'normal':
            mean = np.mean(actual)
            std  = np.std(actual)
            theoretical = np.random.normal(
                loc=mean, scale=std, size=len(actual)
            )
        elif theoretical == 'uniform' or theoretical == 'random':
            theoretical = np.random.uniform(
                np.min(actual), np.max(actual), len(actual)
            )
        elif theoretical == 'pvalue' or theoretical == 'p':
            theoretical = np.random.random_sample(len(actual))
        else:
            raise ValueError('Invalid theoretical')

    if kind == 'qq_log':
        actual = -np.log10(actual)
        theoretical = -np.log10(theoretical)
        kind = 'qq'

    reg_pp = True  # If false, do a pp plot that is evenly spaced in the hist.
    if kind is 'lin_pp':
        kind = 'cum'
        reg_pp = False

    # Choose central plot type
    if kind == 'qq':
        cum = False
        if not title:
            title = 'QQ Plot'
        # We use percentiles, so get evenly spaced percentiles from 0% to 100%
        q = np.linspace(0, 100, bins+1)
        xhist = np.percentile(theoretical, q)
        yhist = np.percentile(actual, q)

    elif kind == 'cum':
        cum = True
        # Create bins from sorted data
        theoretical_sort = sorted(theoretical)
        # Bins with approximately equal numbers of points
        if reg_pp:
            boundaries = uniform_bins(theoretical_sort, bins)
            if not title:
                title = 'Cumulative Probability Plot'
        # Bins with equal ranges
        else:
            mx = max(np.max(x), np.max(y))
            mx += mx*0.01
            mn = min(np.min(x), np.min(y))
            mn -= mx*0.01
            boundaries = np.linspace(mn, mx, bins+1, endpoint=True)
            if not title:
                title = 'Linear Spaced Cumulative Probability Plot'

        labels = [
            (boundaries[i]+boundaries[i+1])/2 for i in range(len(boundaries)-1)
        ]

        # Bin two series into equal bins
        xb = pd.cut(x, bins=boundaries, labels=labels)
        yb = pd.cut(y, bins=boundaries, labels=labels)

        # Get value counts for each bin and sort by bin
        xhist = xb.value_counts().sort_index(ascending=True)/len(xb)
        yhist = yb.value_counts().sort_index(ascending=True)/len(yb)

        # Make cumulative
        for ser in [xhist, yhist]:
            ttl = 0
            for idx, val in ser.iteritems():
                ttl += val
                ser.loc[idx] = ttl

    # Set labels
    if not xlabel:
        if hasattr(x, 'name'):
            xlabel = x.name
        else:
            xlabel = 'Theoretical'
    if not ylabel:
        if hasattr(y, 'name'):
            ylabel = y.name
        else:
            ylabel = 'Actual'

    # Create figure layout
    if fig or ax:
        fig, ax == _get_fig_ax(fig, ax)
        style = 'simple'

    elif style == 'simple':
        fig, ax = plt.subplots(figsize=(size,size))

    elif style == 'joint':
        # Create a jointgrid
        sns.set_style('darkgrid')
        gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])
        fig = plt.figure(figsize=(size,size))
        ax  = plt.subplot(gs[1, 0])
        axt = plt.subplot(gs[0, 0], sharex=ax, yticks=[])
        axr = plt.subplot(gs[1, 1], sharey=ax, xticks=[])
        # Plot side plots
        axt.hist(xhist, bins=bins, cumulative=cum)
        axr.hist(yhist, bins=bins, cumulative=cum, orientation='horizontal')

    elif style == 'column':
        # Create a two column grid
        fig = plt.figure(figsize=(size*2,size))
        ax  = plt.subplot2grid((2,2), (0,0), rowspan=2)
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax3 = plt.subplot2grid((2,2), (1,1), sharex=ax2, sharey=ax2)
        # Plot extra plots - these ones are traditional histograms
        sns.distplot(actual, ax=ax2, bins=bins)
        sns.distplot(theoretical, ax=ax3, bins=bins)
        ax2.set_title(ylabel)
        ax3.set_title(xlabel)
        for a in [ax2, ax3]:
            a.set_frame_on(True)
            a.set_xlabel = ''
            a.axes.get_yaxis().set_visible(True)
            a.yaxis.tick_right()
            a.yaxis.set_label_position('right')
            a.yaxis.set_label('count')

    # Plot the scatter plot
    ax.scatter(xhist, yhist, label='')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    # Make the plot a square
    emin = min(np.min(xhist), np.min(yhist))
    emax = max(np.max(xhist), np.max(yhist))
    t2b = abs(emax-emin)
    scale = t2b*0.01
    emin -= scale
    emax += scale
    lim = (emin, emax)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # Plot a 1-1 line in the background
    ax.plot(lim, lim, '-', color='0.75', alpha=0.9, zorder=0.9)

    # Add Mann-Whitney U p-value
    mwu   = sts.mannwhitneyu(actual, theoretical)
    chi = sts.chisquare(yhist, xhist)
    handles, _ = ax.get_legend_handles_labels()
    prc, prp = sts.pearsonr(xhist, yhist)
    if round(prc, 4) > 0.998:
        plbl = 'pearsonr = {:.2f}; p = {:.2f}'
    else:
        plbl = 'pearsonr = {:.2}; p = {:.2e}'
    handles.append(
        mpatches.Patch(
            color='none', label=plbl.format(prc, prp)
        )
    )
    if kind == 'qq' or reg_pp:
        if chi.pvalue < 0.001:
            cpl = 'chisq = {:.2}; p = {:2e}'
        else:
            cpl = 'chisq = {:.2}; p = {:.2f}'
        handles.append(
            mpatches.Patch(
                color='none', label=cpl.format(chi.statistic, chi.pvalue)
            )
        )
    if mwu.pvalue < 0.001:
        mwl = 'mannwhitneyu = {:.2}; p = {:.2e}'
    else:
        mwl = 'mannwhitneyu = {:.2}; p = {:.5f}'
    handles.append(
        mpatches.Patch(
            color='none', label=mwl.format(mwu.statistic, mwu.pvalue)
        )
    )

    ax.legend(handles=handles, loc=0)

    fig.tight_layout()

    if title:
        if style == 'simple':
            ax.set_title(title, fontsize=14)
        else:
            fig.suptitle(title, fontsize=16)
        if style == 'joint':
            fig.subplots_adjust(top=0.95)
        elif style != 'simple':
            fig.subplots_adjust(top=0.93)

    # Last thing is to set the xtick labels for cummulative plots
    if kind == 'cum':
        labels = [round(i, 2) for i in labels]
        fig.canvas.draw()
        xlabels = ax.get_xticklabels()
        ylabels = ax.get_yticklabels()
        for i, tlabels in enumerate([xlabels, ylabels]):
            for tlabel in tlabels:
                pos = tlabel.get_position()
                if round(pos[i], 2) < 0.0 or round(pos[i], 2) > 1.0:
                    tlabel.set_text('')
                    continue
                if round(pos[i], 2) == 0.00:
                    txt = labels[0]
                elif round(pos[i], 2) == 1.00:
                    # The last label is the end of the bin, not the start
                    txt = round(theoretical_sort[-1], 2)
                else:
                    txt = labels[int(round(len(labels)*pos[i]))]
                tlabel.set_text(str(txt))
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

    if style == 'joint':
        ax = (ax, axt, axr)
    elif style == 'column':
        ax = (ax, ax2, ax3)

    return fig, ax


def scatter(x, y, df=None, xlabel=None, ylabel=None, title=None, pval=None,
            density=True, log_scale=False, handle_nan=True, regression=True,
            fill_reg=True, reg_details=False, labels=None, label_lim=10,
            shift_labels=False, highlight=None, highlight_label=None,
            legend='best', add_text=None, scale_factor=0.05, size=10,
            cmap=None, cmap_midpoint=0.5, lock_axes=True, fig=None, ax=None):
    """Create a simple 1:1 scatter plot plus regression line.

    Always adds a 1-1 line in grey and a regression line in green.

    Can color the points by density if density is true (otherwise they are
    always blue), can also do regular or negative log scaling.

    Regression is done using the statsmodels OLS regression with a constant
    added to the X values using sm.add_constant() to add a column of ones to
    the array of x values. If a constant is already present none is added.

    Parameters
    ----------
    x : Series or str if df provided
        X values
    y : Series or str if df provided
        Y values
    df : DataFrame, optional
        If provided, x and y must be strings that are column names
    {x,y}label : str, optional
        A label for the axes, defaults column value if df provided
    title : str, optional
        Name of the plot
    pval : float, optional
        Draw a line at this point*point count
    density : bool or str, optional
        Color points by density, 'kde' uses a cool kde method, but is too slow
        on large datasets
    log_scale : str, optional
        Plot in log scale, can also be 'negative' for negative log scale.
    handle_nan : bool, optional
        When converting data to log, drop nan and inf values
    regression : bool, optional
        Do a regression
    fill_reg : bool, optional
        Add confidence lines to regression line
    reg_details : bool, optional
        Print regression summary
    labels : Series, optional
        Labels to show, must match indices of plotted data
    label_lim : int, optional
        Only show top # labels on each side of the line
    shift_labels: bool, optional
        If True, try to spread labels out. Imperfect.
    highlight_label : Series, optional
        Boolean series of same len as x/y
    legend : str, optional
        The location to place the legend
    add_text : str, optional
        Text to add to the legend
    scale_factor : float, optional
        A ratio to expand the axes by.
    size : int or tuple, optional
        Size of figure, defaults to square unless (x, y) length tuple given
    cmap : str or cmap, optional
        A cmap to use for density, defaults to a custom blue->red, light->dark
        cmap
    cmap_midpoint : float 0 <= n <= 1, optional
        A midpoint for the cmap, 0.5 means no change
        emphasizes density
    lock_axes : bool, optional
        Make X and Y axes the same length
    fig/ax : matplotlib objects, optional
        Use these instead

    Returns
    -------
    fig : plt.figure
    ax : plt.axes
    reg : plots.LinearRegression or None
        Statsmodel OLS regression results wrapped in a LinearRegression
        object. If no regression requested, returns None
    """
    f, a = _get_fig_ax(fig, ax, size=size)
    #  a.grid(False)
    if isinstance(df, pd.DataFrame):
        assert isinstance(x, str)
        assert isinstance(y, str)
        if log_scale:
            df = df[(df[x] > 0) & (df[y] > 0)]
        if not xlabel:
            xlabel = x
        x = df[x]
        if not ylabel:
            ylabel = y
        y = df[y]
    elif df is not None:
        raise ValueError('df must be a DataFrame or None')

    if not xlabel and hasattr(x, 'name'):
        xlabel = x.name
    if not ylabel and hasattr(y, 'name'):
        ylabel = y.name

    if not xlabel:
        xlabel = 'X'
    if not ylabel:
        ylabel = 'Y'

    if hasattr(x, 'astype'):
        x = x.astype(np.float)
    else:
        x = np.float(x)
    if hasattr(y, 'astype'):
        y = y.astype(np.float)
    else:
        y = np.float(y)

    # Get a color iterator
    c = iter(sns.color_palette())

    # Set up log scaling if necessary
    if log_scale:
        if log_scale == 'reverse' or log_scale == 'n' or log_scale == 'r':
            log_scale = 'negative'
        if log_scale == '-':
            log_scale = 'negative'
        lx = np.log10(x)
        ly = np.log10(y)
        if handle_nan:
            tdf = pd.DataFrame([x, y, lx, ly]).T
            tdf.columns = ['x', 'y', 'lx', 'ly']
            tdf = tdf.replace([np.inf, -np.inf], np.nan).dropna()
            sys.stderr.write(
                'Dropped {0} nan/inf vals from data\n'.format(len(x)-len(tdf))
            )
            x, y, lx, ly = tdf.x, tdf.y, tdf.lx, tdf.ly
            x.name = xlabel
            y.name = ylabel
            lx.name = xlabel
            ly.name = ylabel
        # Get limits
        xlim, ylim, mlim = get_limits(lx, ly, scale_factor=scale_factor)
        xlim = (10**xlim[0], 10**xlim[1])
        ylim = (10**ylim[0], 10**ylim[1])
        mlim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        # Do the regression
        if regression:
            reg = LinearRegression(lx, ly)
    # No log
    else:
        # Get limits
        xlim, ylim, mlim = get_limits(x, y, scale_factor=scale_factor)
        # Do the regression
        if regression:
            reg = LinearRegression(x, y)

    # If we are locking the axes, then all limits are the same
    if lock_axes:
        xlim = mlim
        ylim = mlim

    # Plot the regression on top
    if regression:
        if reg_details:
            reg.print_reg_summary()
        reg.plot_reg_line(ax=a, unlog=log_scale)
        if fill_reg:
            #  reg.plot_ci_line(ax=a, unlog=log_scale)
            reg.plot_pred_line(ax=a, unlog=log_scale)
    else:
        reg = None

    # Plot pval line
    if pval:
        pval = float(pval)
        pval = pval/len(x)
        if pval >= mlim[0]:
            a.plot(mlim, (pval, pval), color='0.5', alpha=0.3)
            a.plot((pval, pval), mlim, color='0.6', alpha=0.5)
            if log_scale == 'negative':
                mxa = mlim[1]+1.25
                tpos = 10**(mxa)
            elif log_scale:
                mna = mlim[0]-1.25
                tpos = 10**(mna)
            else:
                tpos = mlim[0] - (mlim[0] * .1)
            a.text(tpos, pval, 'p={:.1e}'.format(pval))
            dr = pd.concat([pd.Series(x), pd.Series(y)], axis=1)
            dr.columns = ['x', 'y']
            dr = dr[(dr.x <= pval) | (dr.y <= pval)]
            both = dr[(dr.x <= pval) & (dr.y <= pval)]
            dr = dr[(dr.x <= pval) ^ (dr.y <= pval)]
            a.plot(dr.x, dr.y, 'o', c=sns.xkcd_rgb['gold'], alpha=0.2,
                label='goaway')
            a.plot(both.x, both.y, 'o', c=sns.xkcd_rgb['lilac'], alpha=0.2,
                label='goaway')

    # Actual scatter plot
    scatter_args = dict(
        marker='.', edgecolor='', label=None, picker=True, alpha=0.5
    )
    if density:
        if not cmap:
            cmap = 'plasma'
        if cmap == 'spectrum':
            density_colors = dict(
                n_colors=16, start=2.0, rot=1.8, light=0.65, dark=0.3
            )
            cmap = sns.cubehelix_palette(**density_colors, as_cmap=True)
        elif cmap == 'purple':
            density_colors = dict(
                n_colors=16, start=1.8, rot=0.1
            )
            cmap = sns.cubehelix_palette(**density_colors, as_cmap=True)
        elif isinstance(cmap, str):
            cmap = getattr(mpl.cm, cmap)
        if round(cmap_midpoint, 2) != 0.5:
            cmap = _shift_cmap(cmap, midpoint=cmap_midpoint)
        # Use kde of 10k points
        if log_scale:
            cols = [lx, ly, x, y]
            coll = [xlabel, ylabel, 'orig_x', 'orig_y']
        else:
            cols = [x, y]
            coll = [xlabel, ylabel]
        cdf = pd.DataFrame(cols).T
        cdf.columns = coll

        # Plot density using estimate
        if len(cdf) > 0:
            cx, cy = cdf[xlabel], cdf[ylabel]
            # Plot remainder with fast kde
            grid, extents, cz = fast_kde(cx, cy, sample=True)
            idx = cz.argsort()
            if 'orig_x' in cdf.columns:
                x3, y3, z3 = cdf['orig_x'][idx], cdf['orig_y'][idx], cz[idx]
            else:
                x3, y3, z3 = cx[idx], cy[idx], cz[idx]
            #f.colorbar(grid, ax=a, orientation='vertical', shrink=0.75, pad=0.05)
            a.scatter(
                x3, y3, c=z3, cmap=cmap, zorder=2, **scatter_args
            )

    else:
        # Plot the points as blue dots
        a.scatter(
            x, y, color=cmap.colors[0], zorder=2, **scatter_args
        )

    if highlight is not None:
        highlight = pd.Series(highlight).tolist()
        x3 = pd.Series(x)[highlight]
        y3 = pd.Series(y)[highlight]
        a.plot(
            x3, y3, 'o', color=next(c), alpha=0.4, label='  ' +
            highlight_label
        )

    # Plot a 1-1 line in the background
    one_line = (max(xlim[0], ylim[0]), min(xlim[1], ylim[1]))
    a.plot(one_line, one_line, '-', color='0.9', zorder=4)

    # Apply log prior to adding text and labels
    if log_scale:
        a.loglog()

    handles, labls = a.get_legend_handles_labels()
    rm = []
    for i, labl in enumerate(labls):
        if labl == 'goaway':
            rm.append(handles[i])
    for rmthis in rm:
        handles.remove(rmthis)

    if add_text:
        handles.append(
            mpatches.Patch(color='none', label=add_text)
        )

    a.legend(
        handles=handles,
        loc=0, fancybox=True, fontsize=10,
        handlelength=0, handletextpad=0
    )

    if labels is not None:
        # Label most different dots
        text = get_labels(labels, x, y, label_lim, log_scale)
        text = [a.text(*i) for i in set(text)]
        if shift_labels:
            if log_scale:
                adjust_text(
                    text, ax=a, text_from_points=True, expand_text=(0.1, .15),
                    expand_align=(0.15, 0.8), expand_points=(0.1, 10.9),
                    draggable=True,
                    arrowprops=dict(arrowstyle="->", color=next(c), lw=0.5)
                )
            else:
                adjust_text(
                    text, ax=a, text_from_points=True,
                    arrowprops=dict(arrowstyle="->", color=next(c), lw=0.5)
                )

    # Box in the axes
    a.set_xlim(xlim)
    a.set_ylim(ylim)

    # Invert zxes if necessary
    if log_scale == 'negative':
        a.invert_xaxis()
        a.invert_yaxis()

    # Set labels, title, and legend location
    if xlabel:
        a.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        a.set_ylabel(ylabel, fontsize=15)
    if title:
        a.set_title(title, fontsize=20)

    plt.xticks(rotation=0)

    return f, a, reg


###############################################################################
#                               Compound Plots                                #
###############################################################################


def compare_overlapping_dfs(df1, df2, labels, col=None, orient='vertical'):
    """Make venn diagram, QQ plot, and scatter plot from two DataFrames.

    Params
    ------
    df{1/2} : DataFrames with identical(ish) indices
    labels : (str, str)
        Labels for the two dfs
    col : str, optional
        Optional column to compare, will plot a qq plot alongside the
        venn diagram
    orient : {vertical, horizontal}, optional
        Stack graphs or place next to each other, if plotting two plots

    Returns
    -------
    comb : pandas.core.DataFrame
        Combined dataframe
    fig : plt.figure
    ax : plt.axes
    """
    if col is None:
        sns.set_style('white')
        fig, ax = plt.subplots(figsize=(12,12))
        venn_ax = ax
    else:
        if orient == 'vertical' or orient == 'v':
            fig, ax = plt.subplots(
                2, 1, figsize=(12, 24), sharex=False, sharey=False
            )
        elif orient == 'horizontal' or orient == 'h':
            fig, ax = plt.subplots(
                1, 2, figsize=(12, 6), sharex=False, sharey=False
            )
        else:
            raise ValueError("orient must be one of {'vertical', 'horizontal'}")
        venn_ax = ax[0]
        sax = ax[1]

    fig, venn_ax, venn = venn_diagram(
        [df1.index.to_series(), df2.index.to_series()],
        labels=labels, fig=fig, ax=venn_ax
    )

    # Combine the dfs
    suf = ['_{0}'.format(i) for i in labels]
    df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=suf)

    # Do a scatter plot if requested
    if col is not None:
        ylabel, xlabel = ['{0}_{1}'.format(col, i) for i in labels]
        fig, sax = distcomp(
            df2[col], df1[col], bins=1000, kind='qq', style='simple',
            fig=fig, ax=sax, xlabel=xlabel, ylabel=ylabel,
        )
        sns.despine(fig, sax)
        venn_ax.set_title('Venn')
        fig.suptitle('Compare {0} to {1}'.format(*labels), fontsize=20)
        fig.subplots_adjust(top=0.80)

    return df, fig, ax, venn


###############################################################################
#                                  Box Plots                                  #
###############################################################################


def boxplot(data, ylabel, title, box_width=0.35, log_scale=False,
            fig=None, ax=None):
    """Create a formatted box plot.

    From:
        http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/

    Args:
        data (list):       [{label: array}]
        ylabel (str):      A label for the y axis
        title (str):       Name of the plot
        box_width (float): How wide boxes should be, can be 'None' for auto
        log_scale (str):   Plot in log scale, can also be 'negative' for
                           negative log scale.
        fig:               A pyplot figure
        ax:                A pyplot axes object

    Returns:
        (fig, ax): A pyplot figure and axis object in a tuple
    """
    f, a = _get_fig_ax(fig, ax)
    a.set_title(title, fontsize=17)

    # Create lists
    labels = [i for i, j in data]
    pdata  = [i for i in data.values()]

    # Log
    if log_scale:
        a.semilogy()
        if log_scale == 'negative':
            a.invert_yaxis()

    # Plot the box plot
    box_args = dict(
        notch=True,
        bootstrap=10000,
        labels=labels,
        patch_artist=True,
    )
    if box_width:
        box_args.update(dict(widths=0.35))
    bp = a.boxplot(pdata, **box_args)

    # Set Axis Labels
    a.set_ylabel(ylabel, fontsize=15)
    a.set_xticklabels(labels, fontsize=15)
    a.get_xaxis().tick_bottom()
    a.get_yaxis().tick_left()

    # Style plots
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3', linewidth=2)
        # change fill color
        box.set(facecolor='#1b9e77')
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    return f, a


###############################################################################
#                              Specialized Plots                              #
###############################################################################


def venn_diagram(sets, labels=None, fig=None, ax=None, size=12):
    """Draw a venn diagram from either 2 or 3 sets.

    Params
    ------
    sets : list of sets or Series
        List of at least two sets to diagram
    labels : list of str
        List of same length as sets, names for sets

    Returns
    -------
    fig, ax, VennDiagram
    """
    f, a = _get_fig_ax(fig, ax, size=size)
    if labels:
        assert len(labels) == len(sets)
    else:
        labels = []
        for i, s in enumerate(sets):
            if hasattr(s, 'name'):
                labels.append(s.name)
            else:
                labels.append('Set_{0}'.format(i))
    fsets = []
    for s in sets:
        fsets.append(set(s))
    if len(sets) == 2:
        venn = matplotlib_venn.venn2(fsets, set_labels=labels, ax=a)
    elif len(sets) == 3:
        venn = matplotlib_venn.venn3(fsets, set_labels=labels, ax=a)
    else:
        raise ValueError('Can only handle sets of two or three.')

    return f, a, venn


def manhattan(chrdict, title=None, xlabel='genome', ylabel='values',
              colors='bgrcmyk', log_scale=False, line_graph=False,
              sig_line=None, ax=None):
    """Plot a manhattan plot from a dictionary of 'chr'->(pos, #).

    https://github.com/brentp/bio-playground/blob/master/plots/manhattan-plot.py

    Params
    ------
    chrdict : dict
        A dictionary of {'chrom': [(position, p-value),..]}
    title : str, optional
        A title for the plot
    {x/y}label : str, optional
        Optional label for y/x axis.
    colors : str, optional
        A string of colors (described below) to alternate through while
        plotting different chromosomes, colors:
            b: blue
            g: green
            r: red
            c: cyan
            m: magenta
            y: yellow
            k: black
            w: white
    log_scale : bool, optional
        Use a log scale for plotting, if True, -log10 is used for p-value
        scaling otherwise raw values will be used, should be used for p-values
        only.
    sigline : float, optional
        The point at which to plot the significance line, it is corrected for
        multiple testing by dividing it by the number of tests. Default is
        None, which means no line is plotted.
    line_graph : bool, optional
        If line_graph is True, the data will be plotted as lines instead of
        a scatter plot (not recommended).
    ax : matplotlib ax object, optional
        Pre-created ax object, should be initialized as:
            figure.add_axes((0.1, 0.09, 0.88, 0.85))

    Returns
    --------
    plot : figure or ax object
        An ax object with the plot if ax is provided, otherwise a figure
    """

    xs = []
    ys = []
    cs = []
    colors = cycle(colors)
    xs_by_chr = {}

    last_x = 0
    # Convert the dictionary to a list of tuples sorted by chromosome and
    # positon. Sorted as numbered chomsomes, X, Y, MT, other
    data = sorted(_dict_to_list(chrdict), key=_chr_cmp)

    # Loop through one chromosome at a time
    for chrmid, rlist in groupby(data, key=itemgetter(0)):
        color = next(colors)
        rlist = list(rlist)
        region_xs = [last_x + r[1] for r in rlist]
        xs.extend(region_xs)
        ys.extend([r[2] for r in rlist])
        cs.extend([color] * len(rlist))

        # Create labels for chromsomes that is centered on the middle of the
        # chromsome region on the graph
        xs_by_chr[chrmid] = (region_xs[0] + region_xs[-1]) / 2

        # keep track so that chrs don't overlap.
        last_x = xs[-1]

    xs_by_chr = [(k, xs_by_chr[k]) for k in sorted(xs_by_chr.keys(),
                                                   key=_chr_cmp)]

    # Convert the data into numpy arrays for use in plotting
    xs = np.array(xs)
    ys = -np.log10(ys) if log_scale else np.array(ys)

    # Get the matplotlib object
    retfig = False
    if not ax:
        retfig = True
        f = plt.figure()
        ax = f.add_axes((0.1, 0.09, 0.88, 0.85))  # Define axes boundaries

    # Set a title
    if title:
        plt.title(title)

    # Labels
    ylabel_scale = ' (-log10)' if log_scale else ' (raw)'
    ylabel = ylabel + ylabel_scale
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Actually plot the data
    if line_graph:
        ax.vlines(xs, 0, ys, colors=cs, alpha=0.5)
    else:
        ax.scatter(xs, ys, s=2, c=cs, alpha=0.8, edgecolors='none')

    # plot significance line after multiple testing.
    if sig_line:
        sig_line = sig_line/len(data)
        if log_scale:
            sig_line = -np.log10(sig_line)
        ax.axhline(y=sig_line, color='0.5', linewidth=2)

    # Plot formatting
    ymax = np.max(ys)
    if sig_line:
        ymax = max(ymax + ymax*0.1, sig_line + sig_line*0.1)
    else:
        ymax += ymax*0.1
    plt.axis('tight')  # Puts chromsomes right next to each other
    plt.xlim(0, xs[-1])  # Eliminate negative axis and extra whitespace
    plt.ylim(0, ymax)  # Eliminate negative axis
    plt.xticks([c[1] for c in xs_by_chr],  # Plot the chromsome labels
               [c[0] for c in xs_by_chr],
               rotation=-90, size=8.5)

    # Return a figure if axes were not provided
    if retfig:
        return f
    return ax


###############################################################################
#                              Private Functions                              #
###############################################################################

def fast_kde(x, y, gridsize=(400, 400), extents=None, weights=None,
             sample=False):
    """
    Performs a gaussian kernel density estimate over a regular grid.

    Uses a convolution of the gaussian kernel with a 2D histogram of the data.
    This function is typically several orders of magnitude faster than
    scipy.stats.kde.gaussian_kde for large (>1e7) numbers of points and
    produces an essentially identical result.

    Written by Joe Kington, available here:
        https://gist.github.com/joferkington/d95101a61a02e0ba63e5

    Params
    ------
    x: array-like
        The x-coords of the input data points
    y: array-like
        The y-coords of the input data points
    gridsize: tuple, optional
        An (nx,ny) tuple of the size of the output
        grid. Defaults to (400, 400).
    extents: tuple, optional
        A (xmin, xmax, ymin, ymax) tuple of the extents of output grid.
        Defaults to min/max of x & y input.
    weights: array-like or None, optional
        An array of the same shape as x & y that weighs each sample (x_i,
        y_i) by each value in weights (w_i).  Defaults to an array of ones
        the same size as x & y.
    sample: boolean
        Whether or not to return the estimated density at each location.
        Defaults to False

    Returns
    -------
    density : 2D array of shape *gridsize*
        The estimated probability distribution function on a regular grid
    extents : tuple
        xmin, xmax, ymin, ymax
    sampled_density : 1D array of len(*x*)
        Only returned if *sample* is True.  The estimated density at each
        point.
    """
    #---- Setup --------------------------------------------------------------
    x, y = np.atleast_1d([x, y])
    x, y = x.reshape(-1), y.reshape(-1)

    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    nx, ny = gridsize
    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input x & y arrays!')

    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    extents = xmin, xmax, ymin, ymax
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    #---- Preliminary Calculations -------------------------------------------

    # Most of this is a hack to re-implment np.histogram2d using `coo_matrix`
    # for better memory/speed performance with huge numbers of points.

    # First convert x & y over to pixel coordinates
    # (Avoiding np.digitize due to excessive memory usage!)
    ij = np.column_stack((y, x))
    ij -= [ymin, xmin]
    ij /= [dy, dx]
    ij = np.floor(ij, ij).T

    # Next, make a 2D histogram of x & y
    # Avoiding np.histogram2d due to excessive memory usage with many points
    grid = sp.sparse.coo_matrix((weights, ij), shape=(ny, nx)).toarray()

    # Calculate the covariance matrix (in pixel coords)
    cov = _image_cov(grid)

    # Scaling factor for bandwidth
    scotts_factor = np.power(n, -1.0 / 6) # For 2D

    #---- Make the gaussian kernel -------------------------------------------

    # First, determine how big the kernel needs to be
    std_devs = np.diag(np.sqrt(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)
    kern_nx, kern_ny = int(kern_nx), int(kern_ny)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor**2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.sum(kernel, axis=0) / 2.0
    kernel = np.exp(-kernel)
    kernel = kernel.reshape((kern_ny, kern_nx))

    #---- Produce the kernel density estimate --------------------------------

    # Convolve the gaussian kernel with the 2D histogram, producing a gaussian
    # kernel density estimate on a regular grid

    # Big kernel, use fft...
    if kern_nx * kern_ny > np.product(gridsize) / 4.0:
        grid = sp.signal.fftconvolve(grid, kernel, mode='same')
    # Small kernel, use ndimage
    else:
        grid = sp.ndimage.convolve(grid, kernel, mode='constant', cval=0)

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * cov * scotts_factor**2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    if sample:
        i, j = ij.astype(int)
        return grid, extents, grid[i, j]
    else:
        return grid, extents


def _image_cov(data):
    """Efficiently calculate the cov matrix of an image."""
    def raw_moment(data, ix, iy, iord, jord):
        data = data * ix**iord * iy**jord
        return data.sum()

    ni, nj = data.shape
    iy, ix = np.mgrid[:ni, :nj]
    data_sum = data.sum()

    m10 = raw_moment(data, ix, iy, 1, 0)
    m01 = raw_moment(data, ix, iy, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum

    u11 = (raw_moment(data, ix, iy, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, ix, iy, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, ix, iy, 0, 2) - y_bar * m01) / data_sum

    cov = np.array([[u20, u11], [u11, u02]])
    return cov


def _shift_cmap(cmap, start=0, midpoint=0.5, stop=1.0):
    """Function to offset the "center" of a colormap.

    Useful for data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    From: https://stackoverflow.com/questions/7404116

    Params
    ------
    cmap : matplotlib.cm type cmap
        The cmap to be altered
    start : float, optional
        Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower ofset). Should be between
        0.0 and `midpoint`.
    midpoint : float, optional
        The new center of the colormap. Defaults to 0.5 (no shift). Should be
        between 0.0 and 1.0. In general, this should be  1 - vmax/(vmax +
        abs(vmin)) For example if your data range from -15.0 to +5.0 and you
        want the center of the colormap at 0.0, `midpoint` should be set to  1
        - 5/(5 + 15)) or 0.75
    stop : float, optional
        Offset from highets point in the colormap's range.  Defaults to 1.0 (no
        upper ofset). Should be between `midpoint` and 1.0.

    Returns
    -------
    new_cmap : matplotlib.cm type cmap
        Altered cmap
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    name = cmap.name + '_shifted'

    new_cmap = colors.LinearSegmentedColormap(name, cdict)

    return new_cmap


def get_labels(labels, x, y, lim, log_scale):
    """Choose most interesting labels."""
    p = pd.concat([pd.Series(labels).reset_index(drop=True),
                   pd.Series(x).reset_index(drop=True),
                   pd.Series(y).reset_index(drop=True)], axis=1)
    p.columns = ['label', 'x', 'y']
    # Add calculation columns
    p['small'] = p.x*p.y
    p['interesting'] = (
        (p.small.apply(lambda x: 1/x)) *
        10**np.abs(np.log10(p.x) - np.log10(p.y))
    )
    if log_scale:
        p['diff'] = np.log10(p.x) - np.log10(p.y)
        p['adiff'] = np.log10(p.y) - np.log10(p.x)
    else:
        p['diff'] = p.x - p.y
        p['adiff'] = p.y - p.x
    # Get top smallest pvalues first 5% of points
    p = p.sort_values('small', ascending=True)
    labels = pick_top(p, lim*0.05)
    # Get most different and significant 55% of points
    p = p.sort_values('interesting', ascending=False)
    labels += pick_top(p, lim*0.55, labels)
    # Get most above line, 20% of points
    p = p.sort_values('diff', ascending=True)
    labels += pick_top(p, lim*0.2, labels)
    # Get most below line, 20% of points
    p = p.sort_values('adiff', ascending=True)
    labels += pick_top(p, lim*0.2, labels)
    return labels


def pick_top(p, lim, locs=None):
    """Pick top points if they aren't already in locs."""
    locs  = locs if locs else []
    text  = []
    count = 0
    for l in p.index.to_series().tolist():
        loc = (float(p.loc[l]['x']), float(p.loc[l]['y']), p.loc[l]['label'])
        if loc in locs:
            continue
        locs.append(loc)
        text.append(loc)
        count += 1
        if count >= lim:
            break
    return text


def repel_labels(ax, text, k=0.01):
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    for xi, yi, label in text:
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    #  arrowprops=dict(arrowstyle="->",
                                    #  shrinkA=0, shrinkB=0,
                                    #  connectionstyle="arc3",
                                    #  color='red'),
                    )
    # expand limits
    all_pos = np.vstack(pos.values())
    x_span, y_span = np.ptp(all_pos, axis=0)
    mins = np.min(all_pos-x_span*0.15, 0)
    maxs = np.max(all_pos+y_span*0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])


def scale_val(val, factor, direction):
    """Scale val by factor either 'up' or 'down'."""
    if direction == 'up':
        return val+(val*factor)
    if direction == 'down':
        return val-(val*factor)
    raise ValueError('direction must be "up" or "down"')


def scale_rng(rng, factor):
    """Scale (low, hi), by factor in direction (down, up)."""
    if rng[0] > 0:
        fac = (rng[0]-rng[1])*factor
    else:
        fac = (rng[1]-rng[0])*factor
    lw, hi = rng[0]-fac, rng[1]+fac
    # Try to handle infinates by individual scaling
    if abs(lw) == np.inf:
        lw = scale_val(rng[0], factor, 'down')
    if abs(hi) == np.inf:
        hi = scale_val(rng[1], factor, 'up')
    # If still infinate, just use the unscaled value
    if abs(lw) == np.inf:
        sys.stderr.write('Scaled low value was infinate, unscaling\n')
        lw = rng[0]
    if abs(hi) == np.inf:
        sys.stderr.write('Scaled high value was infinate, unscaling\n')
        hi = rng[1]
    return lw, hi


def get_limits(x, y, scale_factor=0.05):
    """Return xlim, ylim, mlim."""
    xlim = (np.min(x), np.max(x))
    ylim = (np.min(y), np.max(y))
    if scale_factor:
        xlim = scale_rng(xlim, factor=scale_factor)
        ylim = scale_rng(ylim, factor=scale_factor)
    mn = min(xlim[0], ylim[0])
    mx = max(xlim[1], ylim[1])
    mlim = (mn, mx)
    return xlim, ylim, mlim


def uniform_bins(seq, bins=100):
    """Returns unique bin edges for an iterable of numbers.

    Note: to make edges unique, drops duplicate edges, for large
    datasets with many duplicates in a skewed distribution, this
    could result in fewer bins than requested.

    Params
    ------
    seq  : iterable of numbers
    bins : int

    Returns
    -------
    edges : list
        A list of edges, the first is the first entry of seq, the last
        is the last entry, all others are approximately evenly sized
    """
    avg = len(seq)/float(bins)
    out = [seq[0]]
    last = 0.0

    while last < len(seq)-avg:
        bin_edge = seq[int(last + avg)]
        if bin_edge not in out:
            out.append(bin_edge)
        last += avg

    # Guarantee that right edge is included
    if seq[-1] not in out:
        out.append(seq[-1])

    return out


def _get_fig_ax(fig, ax, size=9):
    """Check figure and axis, and create if none."""
    if fig and not ax:
        ax = fig.axes
    elif ax and not fig:
        fig = ax.figure
    if fig and ax:
        return fig, ax
    else:
        import seaborn as sns
        sns.set_style('darkgrid')
        if isinstance(size, int):
            size = (size, size)
        return plt.subplots(figsize=size)


def _dict_to_list(chrdict):
    """ Convert a dictionary to an array of tuples """
    output = []
    for chromosome, values in chrdict.items():
        for value in values:
            output.append((chromosome, ) + value)
    return output


def _chr_cmp(keys):
    """ Allow numeric sorting of chromosomes by chromosome number
        If numeric interpretation fails, position that record at -1 """
    key = keys[0].lower().replace("_", "")
    chr_num = key[3:] if key.startswith("chr") else key
    if chr_num == 'x':
        chr_num = 98
    elif chr_num == 'y':
        chr_num = 99
    elif chr_num.startswith('m'):
        chr_num = 100
    else:
        try:
            chr_num = int(chr_num)
        except ValueError:
            chr_num = 101
    return (chr_num, keys[1])
