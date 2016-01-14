"""
============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
   ORIG AUTHOR: Brent Pedersen (University of Utah)
           URL:
https://github.com/brentp/bio-playground/blob/master/plots/manhattan-plot.py

       LICENSE: MIT License
       VERSION: 1.0

       CREATED: 2016-30-13 12:01
 Last modified: 2016-01-13 16:01

   DESCRIPTION: Create a manhattan plot from a dictionary of
                chr->(position, p-value)

         USAGE: import manhattan_plot
                manhattan_plot.plot(dict, sig_line=0.05, title='GWAS',
                                    image_path='./plot.png')

============================================================================
"""
from itertools import groupby, cycle
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np

__all__ = ['plot']

###############################################################################
#                               Primary Function                              #
###############################################################################


def plot(chrdict, sig_line=0.001, title=None, image_path=None,
         colors='bgrcmyk', log_scale=True, line_graph=False):
    """
    Description: Plot a manhattan plot from a dictionary of
                 'chr'->(pos, p-value) with a significance line drawn at
                 the significance point defined by sig_line, which is then
                 corrected for multiple hypothesis testing.

        Returns: A matplotlib.pyplot object

        Options:

    If image_path is given, save at that image (still returns pyplot obj)

    sig_line is the point at which to plot the significance line, it is
             corrected for multiple testing by dividing it by the number
             of tests. Default is 0.001.

    Possible colors for colors string:
                        b: blue
                        g: green
                        r: red
                        c: cyan
                        m: magenta
                        y: yellow
                        k: black
                        w: white

    If log_scale is True, -log10 is used for p-value scaling otherwise
    raw p-values will be used.

    If line_graph is True, the data will be plotted as lines instead of
    a scatter plot (not recommended).

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

    plt.close()  # Make sure we don't overlap the plots
    f = plt.figure()
    ax = f.add_axes((0.1, 0.09, 0.88, 0.85))  # TODO: No idea what this does

    # Set a title
    if title:
        plt.title(title)

    ylabel_scale = ' (-log10)' if log_scale else ' (raw)'
    ylabel = 'p-values' + ylabel_scale
    ax.set_ylabel(ylabel)

    # Actually plot the data
    if line_graph:
        ax.vlines(xs, 0, ys, colors=cs, alpha=0.5)
    else:
        ax.scatter(xs, ys, s=2, c=cs, alpha=0.8, edgecolors='none')

    # plot significance line after multiple testing.
    sig_line = sig_line/len(data)
    if log_scale:
        sig_line = -np.log10(sig_line)
    ax.axhline(y=sig_line, color='0.5', linewidth=2)

    # Plot formatting
    ymax = np.max(ys)
    ymax = max(ymax + ymax*0.1, sig_line + sig_line*0.1)
    plt.axis('tight')  # Puts chromsomes right next to each other
    plt.xlim(0, xs[-1])  # Eliminate negative axis and extra whitespace
    plt.ylim(0, ymax)  # Eliminate negative axis
    plt.xticks([c[1] for c in xs_by_chr],  # Plot the chromsome labels
               [c[0] for c in xs_by_chr],
               rotation=-90, size=8.5)

    # Save if requested
    if image_path:
        plt.savefig(image_path)
    return plt

###############################################################################
#                              Private Functions                              #
###############################################################################


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
