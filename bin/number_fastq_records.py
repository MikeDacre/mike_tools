#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add a #_ to the start of fastq names.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-47-16 20:03
 Last modified: 2016-03-17 10:27

============================================================================
"""
import gzip
import bz2
import sys
import argparse
from Bio.SeqIO.QualityIO import FastqGeneralIterator


###############################################################################
#                                Core Function                                #
###############################################################################


def number_fastqs(infile, outfile):
    """Add a number to the front of the fastq records."""
    count = 0
    with open_zipped(outfile, 'w') as fout:
        for id, seq, q in FastqGeneralIterator(open_zipped(infile)):
            fout.write('@{}:{}\n{}\n+\n{}\n'.format(count, id, seq, q))
            count += 1
    return 0


###############################################################################
#                                House Keeping                                #
###############################################################################


def open_zipped(infile, mode='r'):
    """ Return file handle of file regardless of zipped or not
        Text mode enforced for compatibility with python2 """
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
#                             Running as a script                             #
###############################################################################



def main(argv=None):
    """Run as a script."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional files
    parser.add_argument('-i', '--infile', nargs='?', default=sys.stdin,
                        help="Input file (Default: STDIN)")
    parser.add_argument('-o', '--outfile', nargs='?', default=sys.stdout,
                        help="Output file (Default: STDOUT)")

    args = parser.parse_args(argv)

    return number_fastqs(args.infile, args.outfile)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())

