#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-50-15 12:01
 Last modified: 2016-01-15 13:11

   DESCRIPTION: Convert a simple bed to vcf format.
                Bed is expected to be 0 base with column 4:
                    Black6 allele|CAST allele

                Convert from base 0 to base 1 for vcf
                Create two individuals:
                    BxC -- 1|0
                    CxB -- 0|1

                These can then be used by vcf2diploid to create proper
                maternal and paernal genome files for AlleleSeq

         USAGE: cat bedfile | bed_to_vcf_tomas > vcffile

============================================================================
"""
import sys
import argparse
import gzip
import bz2

VCF_HEADER = """##fileformat=VCFv4.0
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tBxC\tCxB
"""
BXC = '1|0'
CXB = '0|1'
ENDL = '\n'

__all__ = ['bed_to_vcf_tomas']

###############################################################################
#                                Core Function                                #
###############################################################################


def bed_to_vcf_tomas(infile=sys.stdin, outfile=sys.stdout):
    """ Run everything """
    with open_zipped(infile) as fin:
        with open_zipped(outfile) as fout:
            fout.write(VCF_HEADER)
            for i in fin:
                f = i.rstrip().split('\t')
                fout.write('\t'.join([f[0],
                                      f[2],  # Base 1
                                      '.',
                                      '\t'.join(f[3].split('|')),
                                      '.', '.', '.', 'GT',  # Data not needed
                                      BXC, CXB]) +  # BxC and CxB
                           ENDL)


###############################################################################
#                    Private Function: Handle Zipped Files                    #
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
#                            For Running Directly                             #
###############################################################################


def main(argv=None):
    """ Run as a script """
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional files
    parser.add_argument('-b', '--bedfile', nargs='?', default=sys.stdin,
                        help="Input Bed file (Default: STDIN)")
    parser.add_argument('-o', '--outfile', nargs='?', default=sys.stdout,
                        help="Output VCF file (Default: STDOUT)")

    args = parser.parse_args(argv)

    bed_to_vcf_tomas(args.bedfile, args.outfile)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
