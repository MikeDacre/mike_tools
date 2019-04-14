#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pick causal SNPs from dap_ss/torus output.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-56-07 16:09
 Last modified: 2016-09-07 17:55

   DESCRIPTION: Currently just uses a PIP > 0.9 as causal

============================================================================
"""
import sys
import bz2
import gzip
import argparse
import pickle


def pick_causal(infile=sys.stdin, outfile=sys.stdout, exclude=False,
                statfile=sys.stderr):
    """Print a new torus style output with an additional column of causal Y/N.

    Assumes that all PIPs about 0.9 are causal.

    :infile:   A file location to read. STDIN ok.
    :outfile:  A file location to write. STDOUT ok.
    :statfile: A file to write a pickled dictionary of data on peaks with no
               causal SNP.
    :exclude:  Skip singletons and peaks with no causal SNP.
    """
    fin        = iter_file(infile)
    peaks      = 0
    singletons = 0
    no_causal  = {}
    csl        = {}
    with open_zipped(outfile, 'w') as fout:
        for peak, snps in fin:
            peaks += 1
            posteriors = [s[1] for s in snps]
            # Exclude singletons
            if len(posteriors) == 1:
                singletons += 1
                if exclude:
                    continue
            c = 0
            for snp, posterior in snps:
                ############################
                #  Actual algorithm here.  #
                ############################
                causal = 'Y' if posterior > 0.9 else 'N'
                if causal == 'Y':
                    c += 1
                fout.write('\t'.join([snp, peak, '1.7230e-01',
                                      str(posterior), causal]) + '\n')
            if c in csl:
                csl[c] += 1
            else:
                csl[c]  = 1
            if c == 0:
                no_causal[peak] = snps
                if exclude:
                    continue
    with open(statfile, 'wb') as sfile:
        pickle.dump(no_causal, sfile)
    nc = len(no_causal)
    sys.stderr.write('Peaks: {}\n'.format(peaks))
    sys.stderr.write('Causal counts per peak:\n')
    for k, v in csl.items():
        sys.stderr.write('\t{}:\t{}\n'.format(k, v))
    sys.stderr.write('Singletons: {} ({:.2%} of total)\n'
                     .format(singletons, singletons/peaks))
    sys.stderr.write('Non-causal: {} ({:.2%} of total)\n'
                     .format(nc, nc/peaks))
    if exclude:
        sys.stderr.write('Singletons and non-causal excluded from output.\n')


def iter_file(infile):
    """Iterate through the dap output and yield (peak=>[(SNP, prior, post)]).

    :infile: The location of the DAP output file.

    """
    with open_zipped(infile) as bob:
        out   = {}
        nexts = ''
        for line in bob:
            f = line.rstrip().split('\t')
            if not nexts:
                nexts = f[1]
                out[f[1]] = [(f[0], float(f[3]))]
                continue
            if f[1] == nexts:
                out[f[1]].append((f[0], float(f[3])))
                continue
            else:
                j = out.copy()
                out = {f[1]: [(f[0], float(f[3]))]}
                nexts = f[1]
                yield list(j.keys())[0], list(j.values())[0]


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


def main(argv=None):
    """Run as a script."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-x', '--exclude-singletons', action="store_true",
                        help="Exclude all peaks with only one SNP")

    parser.add_argument('-i', '--infile', nargs='?', default=sys.stdin,
                        help="Input file (Default: STDIN)")
    parser.add_argument('-o', '--outfile', nargs='?', default=sys.stdout,
                        help="Output file (Default: STDOUT)")
    parser.add_argument('-s', '--statfile', nargs='?', default='non_causal.txt',
                        help="Write detailed stats on peaks without causal " +
                        "SNP using pickle (Default: non_causal.txt)")

    args = parser.parse_args(argv)

    pick_causal(args.infile, args.outfile, args.exclude_singletons,
                args.statfile)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
