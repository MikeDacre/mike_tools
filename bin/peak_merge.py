#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge peaks in a file by clustering instead of simple overlap merge.

===============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-29-24 16:08
 Last modified: 2016-08-25 18:40

   DESCRIPTION: When calling ATACseq peaks with MACS in different populations,
                a single peak can be shifted slightly in such a way that
                multiple peaks are collapsed into a single peak when using
                bedtools merge.

                To avoid this issue, this script uses information about the
                prior peak as well as the next peak and tries to create
                clusters that way. The default overlap to be within a single
                cluster is 75%, this can be set using the -o flag.

                This script uses a file parsing iterator that can be easily
                modified to use any file format, the default format is bed.

          NOTE: File must contain all peaks and be sorted by coordinate first.

===============================================================================
"""
import sys
import bz2
import gzip
import argparse
from subprocess import check_output
from collections import OrderedDict

from tqdm import tqdm
import logme

logme.MIN_LEVEL = 'debug'


def peak_merge(peak_file, outfile=sys.stdout, overlap=.75, logfile=sys.stderr):
    """Merge peaks.

    :peak_file: A file handle or sequence file, currently bed only. File
                extension used for parsing, gzip or bzip compression OK, file
                handle OK.
    :outfile:   A bed file of merged peaks with the following fields:
                chr, start, end, count, mean_fold_change, mean_log10pval
                count is the number of members that went into the peak.
    :overlap:   The amount a peak can overlap a prior peak before being moved
                into a new cluster.
    :logfile:   A file to contain some summary stats.
    """
    # Make sure overlap is specified
    if not isinstance(overlap, float):
        overlap = .75
        logme.log('Overlap not specified or not float, using .75', 'info')
    # Specify an offset that is subtracted from peaks to avoid clustering of
    # minor overlaps
    offset         = 4
    # Load file generator
    if isinstance(peak_file, str):
        line_count = int(check_output(
            "wc -l data/all_peaks_sorted.bed | sed 's/ .*//'", shell=True)
                         .decode().rstrip())
    else:
        line_count = None
    pfile          = peak_file_parser(peak_file)
    # Create objects to track stats
    lines          = 0
    clusters       = 0
    cluster_sizes  = {}
    extra_pops     = {}
    # First peak
    prior_peak     = next(pfile)
    cluster        = Cluster(prior_peak)
    lines         += 1
    clusters      += 1
    # Current peak, due to the nature of iterators we need to create a lag.
    peak           = next(pfile)
    lines         += 1
    # Use progress bar if outfile not open already.
    piter = pfile if not isinstance(outfile, str) or logme.MIN_LEVEL == 'debug'\
        else tqdm(pfile, total=line_count, unit='lines')
    # Open outfile and run algorithm
    with open_zipped(outfile, 'w') as fout:
        # Loop through peaks
        for next_peak in piter:
            lines += 1
            overlap_prior = peak.start < prior_peak.end - offset
            logme.log('Prior overlap: {}'.format(overlap_prior), 'debug')
            overlap_next = next_peak.start < peak.end - offset
            logme.log('Next overlap: {}'.format(overlap_next), 'debug')

            ######################
            #  Actual Algorithm  #
            ######################

            # Overlap both, decide by overlap amount.
            if overlap_prior and overlap_next:
                prior_overlap_amount = abs(float(prior_peak.end - peak.start) /
                                           float(cluster.len))
                logme.log('Prior overlap amount: {}'
                          .format(prior_overlap_amount), 'debug')
                prior_peak = peak
                # Overlap prior more than 75%, add to prior cluster.
                if prior_overlap_amount > overlap:
                    cluster.add(peak)
                    # Prep for next run
                    prior_peak = peak
                    peak       = next_peak
                    continue
                # Overlap prior less than 75%, make new cluster.
                else:
                    # Stats
                    if cluster.count in cluster_sizes:
                        cluster_sizes[cluster.count] += 1
                    else:
                        cluster_sizes[cluster.count]  = 1
                    diff = len(cluster.pops)-len(set(cluster.pops))
                    if diff:
                        if diff in extra_pops:
                            extra_pops[diff] += 1
                        else:
                            extra_pops[diff]  = 1
                    clusters    += 1
                    # Write cluster and make new one
                    cluster.write(fout)
                    cluster = Cluster(peak)
                    # Prep for next run
                    prior_peak = peak
                    peak       = next_peak
                    continue
            # Overlap only prior, add to cluster.
            elif overlap_prior:
                cluster.add(peak)
                # Prep for next run
                prior_peak = peak
                peak       = next_peak
                continue
            # Overlap next or none, make new cluster.
            else:
                # Stats
                if cluster.count in cluster_sizes:
                    cluster_sizes[cluster.count] += 1
                else:
                    cluster_sizes[cluster.count]  = 1
                diff = len(cluster.pops)-len(set(cluster.pops))
                if diff:
                    if diff in extra_pops:
                        extra_pops[diff] += 1
                    else:
                        extra_pops[diff]  = 1
                clusters    += 1
                # Write cluster and make new one
                cluster.write(fout)
                cluster      = Cluster(peak)
                # Prep for next run
                prior_peak = peak
                peak       = next_peak
                continue

            # Prep for next run
            prior_peak = peak
            peak       = next_peak

        # Last line
        logme.log('Last peak of file: {}_{}'.format(peak.chrom, peak.start),
                  'debug')
        overlap_prior = peak.start - offset < prior_peak.end
        logme.log('Prior overlap: {}'.format(overlap_prior), 'debug')
        if overlap_prior:
            cluster.add(peak)
            prior_peak = peak
        # Overlap next or none, make new cluster.
        else:
            # Stats
            if cluster.count in cluster_sizes:
                cluster_sizes[cluster.count] += 1
            else:
                cluster_sizes[cluster.count]  = 1
            diff = len(cluster.pops)-len(set(cluster.pops))
            if diff:
                if diff in extra_pops:
                    extra_pops[diff] += 1
                else:
                    extra_pops[diff]  = 1
            clusters    += 1
            # Write cluster and make new one
            cluster.write(fout)
            cluster      = Cluster(peak)
        # This is the end so write the last cluster
        # Stats
        if cluster.count in cluster_sizes:
            cluster_sizes[cluster.count] += 1
        else:
            cluster_sizes[cluster.count]  = 1
        diff = len(cluster.pops)-len(set(cluster.pops))
        if diff:
            if diff in extra_pops:
                extra_pops[diff] += 1
            else:
                extra_pops[diff]  = 1
        # Write cluster and make new one
        cluster.write(fout)

        # Done.

    # Print stats
    logfile.write('\n')
    logme.log('Clustering complete,\nstats:', 'info')
    logfile.write('Total lines:\t{}\n'.format(lines) +
                  'Total clusters:\t{}\n'.format(clusters) +
                  'Total clustered:{}\n'.format(
                      sum([k*v for k,v in cluster_sizes.items()])) +
                  'Cluster sizes:\n')
    for k, v in OrderedDict(cluster_sizes).items():
        logfile.write('\t{}:\t{}\n'.format(k, v))
    if extra_pops:
        logfile.write('Extra peaks for a single population in clusters:\n')
        for k, v in OrderedDict(extra_pops).items():
            logfile.write('\t{}:\t{}\n'.format(k, v))
    else:
        logfile.write('No extra peaks in any cluster.\n')


class Cluster(object):

    """A single merged peak."""

    def __init__(self, peak):
        """Create self.

        :peak: A Peak object.
        """
        self.name        = '{}_{}'.format(peak.chrom, peak.start)
        self.peaks       = [peak]
        self.chrom       = peak.chrom
        self.start       = peak.start
        self.end         = peak.end
        self.len         = self.start - self.end
        self.fold_change = peak.fold_change
        self.log10p      = peak.log10p
        self.pops        = [peak.pop]
        self.count       = 1

    def add(self, peak):
        """Add a peak to this cluster.

        :peak: A Peak object.
        """
        self.peaks.append(peak)
        self.end    = max(self.end, peak.end)
        self.len    = self.start - self.end
        self.count += 1
        self.pops.append(peak.pop)
        # Get average fold change and log10p
        self.fold_change = (self.fold_change+peak.fold_change)/2
        self.log10p      = (self.log10p+peak.log10p)/2

    def write(self, outfile):
        """Write self as a line to outfile.

        :outfile: An open filehandle with write mode.
        """
        outfile.write(
            '\t'.join(
                [
                    str(i) for i in [
                        self.chrom, self.start, self.end, self.count,
                        self.fold_change, self.log10p
                    ]
                ]
            )
        )
        outfile.write('\n')


###############################################################################
#                           File handling functions                           #
###############################################################################


class Peak(object):

    """A simple named list to hold peak details."""

    def __init__(self, chrom, start, end, pop, fold, l10p):
        """Create an instance of self.

        :plist: (chr,start,end,pop,fold_change,log10pval).
        """
        self.chrom       = chrom
        self.start       = int(start)
        self.end         = int(end)
        self.pop         = pop.split('_')[0]
        self.fold_change = float(fold)
        self.log10p      = float(l10p)


def peak_file_parser(infile):
    """Iterator that yields (chr,start,end,fold_change,log10pval) per bed line.

    :infile:  A file handle or sequence file, currently bed only. File
              extension used for parsing, gzip or bzip compression OK, file
              handle OK.
              Note: if infile is STDIN, bed is assumed.
    :returns: An iterator.

    """
    # Open file and make sure supported
    pfile        = open_zipped(infile)
    file_parsers = {'bed': bed_file}
    file_parser  = None
    if pfile.name == '<stdin>':
        file_parser = bed_file
    else:
        for n in file_parsers:
            if in_name(pfile, n):
                file_parser = file_parsers[n]
    if not file_parser:
        nm = pfile.name
        pfile.close()
        raise Exception('Unknown file type: {}'.format(nm))

    # Return the iterator
    return file_parser(pfile)


def bed_file(file_handle):
    """Parse an open bed file, use with peak_file().

    :file_handle: An open file handle to a bed file.
    :yields:      Peak object

    """
    for line in file_handle:
        chrom, start, end, pop, _, _, fold, l10p, _ = line.rstrip().split('\t')
        yield Peak(chrom, start, end, pop, fold, l10p)


def in_name(file_handle, search_string):
    """Check for search_string in name of file_handle.

    Splits name into parts by '.' and looks for complete match in those parts.

    For example, 'bed' will match 'file1.something.bed.gz' and 'bob.bed' but
    not 'file.bedfile'.

    :file_handle:   An open file handle.
    :search_string: The string to match.
    :returns:       True or False

    """
    try:
        parts = file_handle.name.split('.')
    except AttributeError as e:
        sys.stderr.write('Infile must be a file handle\n'
                         'Type: {}\nError:\n'.format(type(file_handle)))
        raise e

    return search_string in parts


def open_zipped(infile, mode='r'):
    """Return file handle of file regardless of zipped or not.

    Text mode enforced for compatibility with python2.
    """
    mode   = mode[0] + 't'
    p2mode = mode
    if hasattr(infile, 'write'):
        return infile
    if isinstance(infile, str):
        if infile.endswith('.gz'):
            return gzip.open(infile, mode)
        if infile.endswith('.bz2'):
            if hasattr(bz2, 'open'):
                return bz2.open(infile, mode)
            else:
                return bz2.BZ2File(infile, p2mode)
        return open(infile, p2mode)


###############################################################################
#                            Command line parsing                             #
###############################################################################


def main(argv=None):
    """Command line parsing."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional arguments
    parser.add_argument('-p', '--percent-overlap', metavar='',
                        help=('Overlap percentage to call single cluster,'
                              'default 0.75 (use decimal, e.g. .75 for 75 '
                              'percent.'))
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Verbose output")

    # Files
    parser.add_argument('-i', '--infile', nargs='?', default=sys.stdin,
                        help="Input file (Default: STDIN)")
    parser.add_argument('-o', '--outfile', nargs='?', default=sys.stdout,
                        help="Output file (Default: STDOUT)")
    parser.add_argument('-l', '--logfile', default=sys.stderr,
                        type=argparse.FileType('a'),
                        help="Log file (Default: STDERR)")

    args = parser.parse_args(argv)

    # Take care of logging
    logme.MIN_LEVEL = 'debug' if args.verbose else 'info'
    logme.LOGFILE   = args.logfile

    peak_merge(peak_file=args.infile, outfile=args.outfile,
               overlap=args.percent_overlap, logfile=args.logfile)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
