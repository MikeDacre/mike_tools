#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
# Copyright © Mike Dacre <mike.dacre@gmail.com>
# Distributed under terms of the MIT license
"""
#====================================================================================
#
#          FILE: bed2gtf (python 3)
#        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
#  ORGANIZATION: Stanford University
#       LICENSE: MIT License, Property of Stanford, Use as you wish
#       VERSION: 0.1
#       CREATED: 2014-08-01 14:25
# Last modified: 2014-08-01 15:10
#
#   DESCRIPTION: Convert a bed12 file to a gtf
#
#         USAGE: Pipe tool, use STDIN and STDOUT
#                Can provide source name, see -h for information
#                If you don't provide a source name (eg refseq), will use
#                'unknown'.
#
#====================================================================================
"""
import argparse
from sys import stdin, stderr

# Get commandline arguments

parser = argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-s', '--source-name', dest='source',
                    help="Name of source, e.g. refSeq")

args = parser.parse_args()
source = args.source if args.source else 'unknown'
if source == 'unknown':
    print("No source provided, using 'unknown'", file=stderr)

for i in stdin:
    f = i.rstrip().split('\t')
    lengths = f[10].split(',')
    starts  = f[11].split(',')
    c = 0
    while c < int(f[9]):
        start = int(starts[c]) + int(f[1])
        end   = int(start) + int(lengths[c])
        gene_string = 'gene_id "' + f[3] + '"; transcript_id "' + f[3] + '"; '
        gene_string = gene_string + 'exon_number "' + str(c)
        print(f[0], source, str(start), str(end), '.', f[5], '.', gene_string,
              sep='\t')
        c = c + 1
