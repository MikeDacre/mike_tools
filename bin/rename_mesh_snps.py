#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Take MeSH input file and name SNPs by position in bed file.

===============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-41-30 11:08
 Last modified: 2016-08-30 15:29

   DESCRIPTION: Given a MeSH input file with names formatted as chr.pos, rename
                to be chr.pos_peak#, using peak positions in a bed file and
                the python_bed_lookup library.

         USAGE: rename_mesh_snps.py -i mesh_input -o new_mesh_file bed_file
                cat mesh_file > rename_mesh_snps.py bed_file > new_mesh_file

          NOTE: This script is incredibly slow for large files. It can process
                about 160 snps/second.

===============================================================================
"""
import sys
import bz2
import gzip
import argparse

# Progress bar
from subprocess import check_output
from tqdm import tqdm

# https://github.com/MikeDacre/python_bed_lookup
from bed_lookup import BedFile


def rename_mesh_snps(bed_file, infile=sys.stdin, outfile=sys.stdout):
    """Rename MeSH infile entries from bed_file peaks.

    :bed_file: Location of a bed file.
    :infile:   Location of MeSH input file, filehandle/stdin OK.
    :outfile:  Where to write renamed MeSH file, filehandle/stdout OK.

    """
    bed = BedFile(bed_file)
    # Progress bar
    if isinstance(infile, str):
        line_count = int(check_output(
            "wc -l {} | sed 's/ .*//'".format(infile), shell=True)
                                      .decode().rstrip())
    else:
        line_count = None
    with open_zipped(outfile, 'w') as fout:
        with open_zipped(infile) as fin:
            # Progress bar
            piter = fin if not isinstance(outfile, str) \
                else tqdm(fin, total=line_count, unit=' snps')
            for line in piter:
                fields = line.split('\t')
                # Actually do the lookup here
                chrom, position = fields[0].split('.')
                peak = bed.lookup(chrom, int(position))
                # If lookup fails, skip this SNP
                if peak is None:
                    continue
                # Rename the SNP
                fields[0] += '-{}'.format(peak)
                # Write the output
                fout.write('\t'.join(fields))
    # Done


def open_zipped(infile, mode='r'):
    """Return file handle of file regardless of zipped or not.

    Text mode enforced for compatibility with python2.

    :infile:  File handle, STDIN, or location of any text, gzip, or bzip2 file.
    :mode:    'r': read; 'a': append; 'w': write. No binary mode.
    :returns: Open file handle.

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
    """Command line parsing."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required file
    parser.add_argument('bed_file',
                        help="Bed file with peak positions.")

    # Optional files
    parser.add_argument('-i', '--infile', metavar='', default=sys.stdin,
                        help="MeSH input file (Default: STDIN)")
    parser.add_argument('-o', '--outfile', metavar='', default=sys.stdout,
                        help="Output: renamed MeSH input file (Default: STDOUT)")

    args = parser.parse_args(argv)

    rename_mesh_snps(args.bed_file, args.infile, args.outfile)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
