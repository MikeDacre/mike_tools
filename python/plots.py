"""
A collection of plotting functions to use with pandas, numpy, and pyplot.

       Created: 2016-36-28 11:10
"""
from operator import itemgetter
from itertools import groupby, cycle

import numpy as np
import scipy.stats as sts
from scipy.stats import gaussian_kde
import pandas as pd

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from matplotlib import gridspec
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

import networkx as nx
from adjustText import adjust_text

# Get rid of pandas future warnings
import warnings as _warn
_warn.simplefilter(action='ignore', category=FutureWarning)
_warn.simplefilter(action='ignore', category=UserWarning)
_warn.simplefilter(action='ignore', category=RuntimeWarning)

###############################################################################
#                             Basic Scatter Plots                             #
###############################################################################


def distcomp(y, x='uniform', bins=100, kind='qq', style=None,
             ylabel=None, xlabel=None, title=None, size=10):
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
    x (theoretical|x-axis) : Series or string, optional
        Series of theoretical data, will go on x-axis, can also be one of
        'normal', 'uniform', or 'pvalue', to use a random distribution.
        The only difference between 'uniform' and 'pvalue' is that pvalue
        min=0 and max=1, while the uniform dist min=min(y) and max=max(y)
    bins : int, optional
        Number of bins to use for plotting, default 1000
    kind : {'qq', 'pp', 'cum', 'lin_pp'}, optional
        qq:  Plot a Q-Q plot
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
    if kind not in ['qq', 'cum', 'pp', 'lin_pp']:
        raise ValueError('kind must be one of qq, pp, cum, or lin_pp')

    if not style:
        if kind is 'qq':
            style = 'simple'
        else:
            style = 'joint'

    if style not in ['simple', 'joint', 'column', 'scatter']:
        raise ValueError('style must be one of simple, joint, or colummn')

    # Convert to old names
    theoretical = x
    actual = y

    kind = 'cum' if kind == 'pp' else kind
    style = 'simple' if style == 'scatter' else style

    if isinstance(theoretical, str):
        if theoretical == 'normal':
            theoretical = np.random.standard_normal(len(actual))
        elif theoretical == 'uniform' or theoretical == 'random':
            theoretical = np.random.uniform(
                np.min(actual), np.max(actual), len(actual)
            )
        elif theoretical == 'pvalue' or theoretical == 'p':
            theoretical = np.random.random_sample(len(actual))
        else:
            raise ValueError('Invalid theoretical')

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

    xlabel = xlabel if xlabel else 'Theoretical'
    ylabel = ylabel if ylabel else 'Actual'

    # Create figure layout
    if style == 'simple':
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
        fig.suptitle(title, fontsize=16)
        if kind == 'joint':
            fig.subplots_adjust(top=0.95)
        else:
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
            labels=None, fig=None, ax=None, density=True, log_scale=False,
            legend='best', label_lim=10, shift_labels=False, highlight=None,
            highlight_label=None, add_text=None, reg_details=True,
            scale_factor=0.02, regression=True, size=10):
    """Create a simple 1:1 scatter plot plus regression line.

    Always adds a 1-1 line in grey and a regression line in green.

    Can color the points by density if density is true (otherwise they are
    always blue), can also do regular or negative log scaling.

    Density defaults to true, it can be fairly slow if there are many points.

    Parameters
    ----------
    x : Series or str if df provided
        X values
    y : Series or str if df provided
        Y values
    df : DataFrame, optional
        If provided, x and y must be strings that are column names
    xlabel : str, optional
        A label for the x axis, defaults to x if df provided
    ylabel : str, optional
        A label for the y axis, defaults to y if df provided
    title : str, optional
        Name of the plot
    pval : float, optional
        Draw a line at this point*point count
    labels : Series, optional
        Labels to show
    density : bool, optional
        Color points by density
    log_scale : str, optional
        Plot in log scale, can also be 'negative' for
                        negative log scale.
    legend : str, optional
        The location to place the legend
    label_lim : int , optional
        Only show top # labels on each side of the line
    shift_labels:    If True, try to spread labels out. Imperfect.
    highlight_label : Series
        Boolean series of same len as x/y
    reg_details : bool
        Print regression summary
    regression : bool
        Do a regression
    scale_factor : float
        A ratio to expand the axes by.
    size : int or tuple
    fig : pyplot figure, optional
    ax : pyplot axes object, optional

    Returns
    -------
    (fig, ax): A pyplot figure and axis object in a tuple
    """
    f, a = _get_fig_ax(fig, ax, size=size)
    #  a.grid(False)
    if df:
        assert isinstance(x, str)
        assert isinstance(y, str)
        if log_scale:
            df = df[(df[x] > 0) & (df[y] > 0)]
        if not xlabel:
            xlabel = x
        x = df[x]
        if not xlabel:
            xlabel = y
        y = df[y]

    # Set up log scaling if necessary
    if log_scale:
        lx = np.log10(x)
        ly = np.log10(y)
        mx = max(np.max(lx), np.max(ly))
        mn = min(np.min(lx), np.min(ly))
        if scale_factor:
            scale = abs(mx-mn)*scale_factor
            mxs = 10**(mx+scale)
            mns = 10**(mn-scale)
        else:
            mxs = 10**mx
            mns = 10**mn
        mlim = (mns, mxs)
        # Do the regression
        if regression:
            model = sm.OLS(ly, lx)
            res = model.fit()
            if reg_details:
                print(res.summary())
            _, iv_l, iv_u = wls_prediction_std(res)
            reg_line = 10**res.fittedvalues
            reg_line_upper = 10**iv_u
            reg_line_lower = 10**iv_l
    # No log
    else:
        mx = max(np.max(x), np.max(y))
        mn = min(np.min(x), np.min(y))
        if scale_factor:
            scale = abs(mx-mn)*scale_factor
            mlim = (mn+scale, mx+scale)
        else:
            mlim = (mn, mx)
        # Do the regression
        if regression:
            model = sm.OLS(y, x)
            res = model.fit()
            P = res.pvalues.tolist()[0]
            if reg_details:
                print(res.summary())
            _, iv_l, iv_u = wls_prediction_std(res)
            reg_line = res.fittedvalues
            reg_line_upper = iv_u
            reg_line_lower = iv_l

    # Plot the regression on top
    if regression:
        inf = 'OLS: {:.3} +/- {:.2}\n    P: {:.2e}\n  $R^2$: {:.2}'.format(
            res.params.tolist()[0], res.bse.tolist()[0],
            res.pvalues.tolist()[0], res.rsquared
        )
        a.plot(x, reg_line, label=inf, alpha=0.8, zorder=10)
        a.fill_between(x, 10**iv_u, 10**iv_l, alpha=0.05, interpolate=True, zorder=9)

    # Plot pval line
    if pval:
        pval = float(pval)
        pval = pval/len(x)
        if pval >= mlim[0]:
            a.plot(mlim, (pval, pval), color='0.5', alpha=0.3)
            a.plot((pval, pval), mlim, color='0.6', alpha=0.5)
            if log_scale == 'negative':
                mxa = mx+1.25
                tpos = 10**(mxa)
            elif log_scale:
                mna = mn-1.25
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

    # Density plot
    if density:
        if log_scale:
            i = lx
            j = ly
        else:
            i = x
            j = y
        xy = np.vstack([i, j])
        try:
            z = gaussian_kde(xy)(xy)
            # Sort the points by density, so that the densest points are
            # plotted last
            idx = z.argsort()
            x2, y2, z = x[idx], y[idx], z[idx]
            s = a.scatter(x2, y2, c=z, s=50,
                          cmap=sns.cubehelix_palette(8, as_cmap=True),
                          edgecolor='', label=None, picker=True, zorder=2)
        except (ValueError, np.linalg.linalg.LinAlgError):
            # Too small for density
            s = a.scatter(x, y,
                          cmap=sns.cubehelix_palette(8, as_cmap=True),
                          edgecolor='', label=None, picker=True, zorder=2)
    else:
        # Plot the points as blue dots
        s = a.scatter(x, y,
                      cmap=sns.cubehelix_palette(8, as_cmap=True),
                      edgecolor='', label=None, picker=True, zorder=2)

    if highlight is not None:
        highlight = pd.Series(highlight).tolist()
        x3 = pd.Series(x)[highlight]
        y3 = pd.Series(y)[highlight]
        a.plot(x3, y3, 'o', c=sns.xkcd_rgb['leaf green'], alpha=0.4,
               label='  ' + highlight_label)

    if log_scale:
        a.loglog()

    # Plot a 1-1 line in the background
    print(mlim)
    a.plot(mlim, mlim, '-', color='0.75', zorder=1000)

    handles, labls = a.get_legend_handles_labels()
    rm = []
    for i, labl in enumerate(labls):
        if labl == 'goaway':
            rm.append(handles[i])
    for rmthis in rm:
        handles.remove(rmthis)

    if add_text:
        handles.append(mpatches.Patch(color='none',
                                      label=add_text))

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
                adjust_text(text, ax=a, text_from_points=True,
                            expand_text=(0.1, .15), expand_align=(0.15, 0.8),
                            expand_points=(0.1, 10.9),
                            draggable=True,
                            arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
            else:
                adjust_text(text, ax=a, text_from_points=True,
                            arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    else:
        a.set_xlim(mlim)
        a.set_ylim(mlim)

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

    return f, a


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
    if fig:
        if bool(fig) == bool(ax):
            f, a = (fig, ax)
        else:
            print('You must provide both fig and ax, not just one.')
            raise Exception('You must provide both fig and ax, not just one.')
        return fig, ax
    else:
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
