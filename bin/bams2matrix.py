#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Copyright © Mike Dacre <mike.dacre@gmail.com>
#
# Distributed under terms of the MIT license
"""
#====================================================================================
#
#          FILE: bams2matrix (python 3)
#        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
#  ORGANIZATION: Stanford University
#       LICENSE: MIT License, Property of Stanford, Use as you wish
#       VERSION: 0.1
#       CREATED: 2015-03-03 11:39
# Last modified: 2015-03-03 12:12
#
#   DESCRIPTION: Convert many bams into a single matrix with one individual per
#                row and all SNPs per column
#
#         USAGE: Run as a script or import as a module.  See '-h' or 'help' for usage
#
#====================================================================================
"""
def merge_bams(files):
    """ Take a list of bam files and convert into a matrix.
        See main info for deatils. """
    import logme

    # Create dirctionary for individuals
    inds = {}

    for file in files:
        """ Loop through all files, pull out SNPs, and add to
            inds dictionary """

def _get_args():
    """ Command Line Argument Parsing """
    import argparse, sys

    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional Arguments
    #parser.add_argument('-v', help="Verbose output")

    # Required Arguments
    parser.add_argument('files', nargs='+',
                        help="Bam files")

    return parser

# Run as a script
if __name__ == '__main__':
    parser = _get_args()
    args = parser.parse_args()

    merge_bams(args.files)
