"""
A collection of plotting functions to use with pandas, numpy, and pyplot.

       Created: 2016-36-28 11:10
 Last modified: 2016-10-28 12:04
"""

def scatter(x, y, xlabel, ylabel, title, fig=None, ax=None,
            legend='upper left'):
    """Create a simple 1:1 scatter plot plus regression line.

    Args:
        x (Series):   X values
        y (Series):   Y values
        xlabel (str): A label for the x axis
        ylabel (str): A label for the y axis
        title (str):  Name of the plot
        fig:          A pyplot figure
        ax:           A pyplot axes object
        legend (str): The location to place the legend

    Returns:
        (fig, ax): A pyplot figure and axis object in a tuple
    """
    if fig:
        if bool(fig) == bool(ax):
            f, a = (fig, ax)
        else:
            print('You must provide both fig and ax, not just one.')
            return None, None
    else:
        f, a = plt.subplots(figsize=(9,9))
    # Define the limits of the plot, we want a 1:1 ratio of axes
    mlim = (-1, max(np.max(x), np.max(y))+1)
    a.set_xlim(*mlim)
    a.set_ylim(*mlim)
    # Plot a 1-1 line in the background
    a.plot(mlim, mlim, '-', color='0.75')
    # Plot the points as blue dots
    a.plot(x, y, 'o', color='b')
    # Do the regression
    m, b, r, p, std = sts.linregress(x, y)
    # Plot the regression line ober the top in green
    a.plot(x, m*x + b, '-', color='g',
           label='regression: r={:.2}, p={:.2}'.format(r_value, p_value))
    # Set labels, title, and legend location
    a.set_xlabel(xlabel, fontsize=15)
    a.set_ylabel(ylabel, fontsize=15)
    a.set_title(title, fontsize=20)
    a.legend(loc=legend)
    return f, a


def scatter_mlog(x, y, xlabel, ylabel, title, fig=None, ax=None
                 legend='upper left'):
    """Create a -log10 scaled 1:1 scatter plot plus regression line.

    Args:
        x (Series):   X values
        y (Series):   Y values
        xlabel (str): A label for the x axis
        ylabel (str): A label for the y axis
        title (str):  Name of the plot
        fig:          A pyplot figure
        ax:           A pyplot axes object
        legend (str): The location to place the legend

    Returns:
        (fig, ax): A pyplot figure and axis object in a tuple
    """
    if fig:
        if bool(fig) == bool(ax):
            f, a = (fig, ax)
        else:
            print('You must provide both fig and ax, not just one.')
            return None, None
    else:
        f, a = plt.subplots(figsize=(9,9))
    lx = -np.log10(x)
    ly = -np.log10(y)
    # Plot in negative log space
    a.semilogx()
    a.semilogy()
    a.invert_xaxis()
    a.invert_yaxis()
    # Define the limits of the plot, we want a 1:1 ratio of axes
    mlim = (10, 10**(-max(np.max(lx), np.max(ly))-2))
    a.set_xlim(*mlim)
    a.set_ylim(*mlim)
    # Plot a 1-1 line in the background
    a.plot(mlim, mlim, '-', color='0.75')
    # Plot the points as blue dots
    a.plot(x, y, 'o', color='b')
    # Do the regression
    m, b, r, p, std = sts.linregress(lx, ly)
    # Plot the regression line ober the top in green
    a.plot(x, 10**(m*-lx + b), '-', color='g',
           label='regression: r={:.2}, p={:.2}'.format(r, p))
    # Set labels, title, and legend location
    a.set_xlabel(xlabel, fontsize=15)
    a.set_ylabel(ylabel, fontsize=15)
    a.set_title(title, fontsize=20)
    a.legend(loc=legend)
    return f, a


def scatter_log(x, y, xlabel, ylabel, title, fig=None, ax=None
                 legend='upper left'):
    """Create a log10 scaled 1:1 scatter plot plus regression line.

    Args:
        x (Series):   X values
        y (Series):   Y values
        xlabel (str): A label for the x axis
        ylabel (str): A label for the y axis
        title (str):  Name of the plot
        fig:          A pyplot figure
        ax:           A pyplot axes object
        legend (str): The location to place the legend

    Returns:
        (fig, ax): A pyplot figure and axis object in a tuple
    """
    if fig:
        if bool(fig) == bool(ax):
            f, a = (fig, ax)
        else:
            print('You must provide both fig and ax, not just one.')
            return None, None
    else:
        f, a = plt.subplots(figsize=(9,9))
    lx = np.log10(x)
    ly = np.log10(y)
    # Plot in negative log space
    a.semilogx()
    a.semilogy()
    # Define the limits of the plot, we want a 1:1 ratio of axes
    mlim = (10, 10**(max(np.max(lx), np.max(ly))+2))
    a.set_xlim(*mlim)
    a.set_ylim(*mlim)
    # Plot a 1-1 line in the background
    a.plot(mlim, mlim, '-', color='0.75')
    # Plot the points as blue dots
    a.plot(x, y, 'o', color='b')
    # Do the regression
    m, b, r, p, std = sts.linregress(lx, ly)
    # Plot the regression line ober the top in green
    a.plot(x, 10**(m*lx + b), '-', color='g',
           label='regression: r={:.2}, p={:.2}'.format(r, p))
    # Set labels, title, and legend location
    a.set_xlabel(xlabel, fontsize=15)
    a.set_ylabel(ylabel, fontsize=15)
    a.set_title(title, fontsize=20)
    a.legend(loc=legend)
    return f, a
