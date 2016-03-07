Python libraries not worthy of their own module
===============================================

manhattan_plot
--------------

Plot a dictionary of points + p-values as a manhattan plot, e.g.::

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

A simple function to print a log with a timestamp. Also provides an exception class with logging support.

Use examples:

    from logme import log
    log.MIN_LEVEL = 'warn'
    log('This won't be printed', level='info', min_level=LOG_LEVEL)
  
No output

    log('This will be written to the file', 'mylog.txt', level='warn', min_level=LOG_LEVEL)

Output (appended to mylog.txt):

    20160117 18:22:11.887 | WARNING --> This will be written to the file

Any filehandle can be used instead of 'mylog.txt', if no file is provided, STDERR is used. If STDOUT or STDERR are used, the flag (e.g. 'INFO' or 'CRITICAL') will be colored according to severity. Multi-line logs are indented like this:

    20160117 18:22:11.887 | WARNING --> This will be written to the file
    ----------------------------------> This is another line

A logging object can also be used, in which can only the timestamp will be added, and newlines will not be indented.

Memory Profile
--------------
Simple script to profile memory of a program. Usage::

    import memory_profile
    print(memory_profile.memory())

This script is not written originally by me, but I can't remember where I got it from. If anyone can figure that out, please tell me and I will update the attribution.


Compare Snps to Genome
----------------------
Take a SNP file in either Bed or VCF format and create a list of Chromsome objects, which contain all of the SNPs.
Then parse a list of FastA files and create a list of SeqIO objects.
Finally, compare all SNPs to the equivalent position in the SeqIO objects, and create three lists to describe matches:
    ref, alt, and no match
These lists are added to the Chromosome object and contain the positions of the SNPs that they describe.

mike.py
-------

Misc functions unlikely to be useful by anyone except me.
