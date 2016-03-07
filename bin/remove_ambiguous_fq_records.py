#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Remove any reads with an 'N' character from both pairs.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-06-01 16:03
 Last modified: 2016-03-07 11:39

   DESCRIPTION: Outputs paired end files in matched order.
                Can provide either single end or paired end.
                Output files are the input file name + 'filtered'. e.g.::
                    Input:  -1 in.1.fq.gz -2 in.2.fq.gz
                    Output: in.1.filtered.fq.gz & in.2.filtered.fq.gz

============================================================================
"""
import sys
import argparse
import gzip
import bz2
try:
    from memory_profile import memory
    profile = True
except ImportError:
    pass
    profile = False
from Bio.SeqIO.QualityIO import FastqGeneralIterator


def filter_single(read_file, decompress=False):
    """Remove all reads with 'N's."""
    with open_zipped(output_name(read_file, decompress), 'w') as outf:
        for id, seq, q in FastqGeneralIterator(open_zipped(read_file)):
            if 'N' not in seq:
                outf.write('@{}\n{}\n+\n{}\n'.format(id, seq, q))


def filter_paired(pair1_file, pair2_file, decompress=False):
    """Remove all reads with 'N's."""
    # Open input files
    pair1 = FastqGeneralIterator(open_zipped(pair1_file))
    pair2 = FastqGeneralIterator(open_zipped(pair2_file))
    # Print outputs
    out1 = open_zipped(output_name(pair1_file, decompress), 'w')
    out2 = open_zipped(output_name(pair2_file, decompress), 'w')
    count = 0
    kept  = 0
    filt  = 0
    mem   = 0
    for (id1, seq1, q1), (id2, seq2, q2) in zip(pair1, pair2):
        try:
            assert id1.split('/')[0] == id2.split('/')[0]
        except AssertionError:
            sys.stderr.write('{} and {} are not the same ID\n'.format(id1,
                                                                      id2))
            return(1)
        if 'N' not in seq1 or 'N' not in seq2:
            out1.write('@{}\n{}\n+\n{}\n'.format(id1, seq1, q1))
            out2.write('@{}\n{}\n+\n{}\n'.format(id2, seq2, q2))
        if profile:
            if count == int(1e3):
                if int(memory()) != int(mem):
                    sys.stderr.write('Memory usage: {}\n'.format(memory()))
                    mem = memory()
                count = 0
                kept += 1
            else:
                filt += 1
            count += 1
    sys.stderr.write('Filtered:\t{0}\nKept:\t\t{1}\nMemory:\t\t{2}MB\n'.format(
        filt, kept, mem))
    return 0


def output_name(input, decompress=False):
    """Replace 'fq' or 'fastq' in input with 'filtered.fq'."""
    out = input.split('.')
    if 'gz' in out:
        out.remove('gz')
    elif 'bz2' in out:
        out.remove('bz2')
    file_end = 'fq' if 'fq' in out else 'fastq'
    try:
        ind  = out.index(file_end)
    except ValueError:
        sys.stderr.write('file must have fq or fastq in the name')
        raise
    out[ind] = 'filtered.' + file_end
    return '.'.join(out)


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

    usage  = '%(prog)s [-1 <pair1> -2 <pair2>] [single_end]'
    parser = argparse.ArgumentParser(
        description=__doc__, usage=usage,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Input files
    fastqs = parser.add_argument_group('FastQ files')
    fastqs.add_argument('single_end', nargs='?',
                        help="Input files")

    fastqs.add_argument('-1', dest='pair1',
                        help="Paired end file 1")
    fastqs.add_argument('-2', dest='pair2',
                        help="Paired end file 1")

    parser.add_argument('-d', '--decompress', action="store_true",
                        help="Output a decomressed file even if input is " +
                        "compressed")

    args = parser.parse_args(argv)

    if not args.single_end and not args.pair1 or not args.pair2:
        if not args.pair1 or not args.pair2:
            sys.stderr.write('FastQ file required.\n')
            parser.print_help()
            return 1
    if bool(args.single_end) is bool(bool(args.pair1) or bool(args.pair2)):
        sys.stderr.write('Cannot handle both single and paired end ' +
                         'simultaneously.\n')
        parser.print_help()
        return 1

    if args.single_end:
        return filter_single(args.single_end, args.decompress)
    else:
        return filter_paired(args.pair1, args.pair2, args.decompress)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
