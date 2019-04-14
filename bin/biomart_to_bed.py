#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Copyright © Mike Dacre <mike.dacre@gmail.com>
"""
#====================================================================================
#
#          FILE: biomart_to_bed (python 3)
#        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
#  ORGANIZATION: Stanford University
#       LICENSE: MIT License, Property of Stanford, Use as you wish
#       VERSION: 0.1
#       CREATED: 2015-02-12 11:29
# Last modified: 2015-02-12 14:32                                                    #
#
#   DESCRIPTION: Convert a biomart export file to a BED file
#                Requires a tab delimited export with the columns intact
#
#====================================================================================
"""
import sys

######################
#    Column Titles   #
######################

# Order:
#'chrom'
#'chromStart'
#'chromEnd'
#'name'
#'score'  --- this isn't included, a value of 1000 is used
#'strand'


columns = ('Chromosome name',
           'Chromosome position start (bp)',
           'Chromosome position end (bp)',
           'Variation Name')

def biomart_to_bed(biomart_file=sys.stdin, bed_file=sys.stdout, columns=columns):
    """ Convert a biomart file to a bed file.
        Requires an open input file handle and output file handle """

    biomart_file.seek(0)
    file_columns = biomart_file.readline().rstrip().split('\t')

    c = ()

    for i in columns:
        if not i == 'NA':
            c = c + (file_columns.index(i),)

    for i in biomart_file:
        f = i.rstrip().split('\t')
        print(f[c[0]], f[c[1]], f[c[2]], file=bed_file)

def _get_args():
    """Command Line Argument Parsing"""
    import argparse

    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional Arguments
    #parser.add_argument('-v', help="Verbose output")

    # Optional Files
    parser.add_argument('-i', '--infile', nargs='?', default=sys.stdin,
                        help="Input file, Default STDIN")
    parser.add_argument('-o', '--outfile', nargs='?', default=sys.stdout,
                        help="Output file, Default STDOUT")
    parser.add_argument('-l', '--logfile', nargs='?', default=sys.stderr,
                        type=argparse.FileType('a'),
                        help="Log File, Default STDERR (append mode)")

    return parser

# Main function for direct running
def main():
    """Run directly"""
    # Get commandline arguments
    parser = _get_args()
    args = parser.parse_args()

    if isinstance(args.infile, str):
        infile_string = True
    else:
        infile_string = False

    if isinstance(args.outfile, str):
        outfile_string = True
    else:
        outfile_string = False

    if infile_string:
        with open(args.infile, 'r') as infile:
            if outfile_string:
                with open(args.outfile, 'w') as outfile:
                    biomart_to_bed(infile, outfile)
            else:
                biomart_to_bed(infile, args.outfile)
    else:
        if outfile_string:
            with open(args.outfile, 'w') as outfile:
                biomart_to_bed(args.infile, outfile)
        else:
            biomart_to_bed(args.infile, args.outfile)

# The end
if __name__ == '__main__':
    main()
