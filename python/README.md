Python libraries not worthy of their own module
===============================================

manhattan_plot
--------------

Plot a dictionary of points + p-values as a manhattan plot, e.g.:

  import manhattan_plot
  manhattan_plot.plot(dict, sig_line=0.05, title='GWAS')

Many thanks to [Brent Pedersen](https://github.com/brentp) for the [original script](https://github.com/brentp/bio-playground/blob/master/plots/manhattan-plot.py). This version is just a modification of that code to allow running as a module and easy python2/3 functionality.

An example output is here: [Manhattan_Plot](https://nbviewer.jupyter.org/github/MikeDacre/mike_tools/blob/master/python/Manhattan_Plot.ipynb)

Usage: `plot(chrdict, sig_line=0.001, title=None, image_path=None, colors='bgrcmyk', log_scale=True, line_graph=False)`

Dictionary format: {'chr': (position, p-value)}

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
    raw p-values will be used, there is no good reason not to use -log10.

    If line_graph is True, the data will be plotted as lines instead of
    a scatter plot (not recommended).

![Manhattan Plot in Python Example](http://i.imgur.com/rC3AmgQ.png)

logme
-----

A simple function to print a log with a timestamp

mike.py
-------

Misc functions unlikely to be useful by anyone except me.
