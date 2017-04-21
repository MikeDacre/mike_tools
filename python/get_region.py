#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Take a list of genome positions and return a 15 base region (-7 to +7)

       Created: 2017-03-20 16:21
 Last modified: 2017-03-20 17:44
"""
import os
import bz2
import gzip
from datetime import timedelta as _td

import fyrd
import sys
import argparse
from Bio import SeqIO as seqio

hg19 = "/godot/genomes/human/hg19"


###############################################################################
#                               Core Algorithm                                #
###############################################################################


def get_regions(positions, genome_file, base=0, count=7):
    """Return a list of regions surrounding a position.

    Will loop through each chromosome and search all positions in that
    chromosome in one batch. Lookup is serial per chromosome.

    Args:
        positions (dict):  Dictionary of {chrom->positons}
        genome_file (str): Location of a genome fasta file or directory of
                           files. If directory, file names must be
                           <chrom_name>.fa[.gz]. Gzipped OK.
        base (int):        Either 0 or 1, base of positions in your list
        count (int):       Distance + and - the position to extract

    Returns:
        dict: {chrom->{postion->sequence}}
    """
    # If genome file is a directory, use recursion! Because why not.
    if os.path.isdir(genome_file):
        chroms = positions.keys()
        files = []
        for chrom in chroms:
            files.append(get_fasta_file(genome_file, chrom))
        final = {}
        for chrom, fl in zip(chroms, files):
            final.update(
                get_dinucleotides({chrom: positions[chrom]}, fl, base, count)
            )
        return final

    done = []
    results = {}
    with open_zipped(genome_file) as fasta_file:
        for chrom in seqio.parse(fasta_file, 'fasta'):
            if chrom.id not in positions:
                continue
            else:
                done.append(chrom.id)
                results[chrom.id] = {}
            for pos in positions[chrom.id]:
                ps     = pos-base  # Correct base-1 positions here
                region = seq(chrom[ps-count:ps+count+1])
                results[chrom.id][pos] = region
    if len(done) != len(positions.keys()):
        print('The following chromosomes were not in files: {}'
              .format([i for i in positions if i not in done]))

    return results


###############################################################################
#                               Parallelization                               #
###############################################################################


def get_regions_parallel(positions, genome_file, base=0, count=7):
    """Return a list of regions surrounding a position.

    Will loop through each chromosome and search all positions in that
    chromosome in one batch. Lookup is serial per chromosome.

    Args:
        positions (dict):  Dictionary of {chrom->positons}
        genome_file (str): Location of a genome fasta file or directory of
                           files. If directory, file names must be
                           <chrom_name>.fa[.gz]. Gzipped OK.
        base (int):        Either 0 or 1, base of positions in your list
        count (int):       Distance + and - the position to extract

    Returns:
        dict: {chrom->{postion->sequence}}
    """
    outs = []
    for chrom in positions.keys():
        if os.path.isdir(genome_file):
            fa_file = get_fasta_file(genome_file, chrom)
        if not os.path.isfile(fa_file):
            raise FileNotFoundError('{} not found.'.format(genome_file))
        mins = int(len(positions[chrom])/2000)+60
        time = str(_td(minutes=mins))
        outs.append(
            fyrd.submit(
                get_regions,
                ({chrom: positions[chrom]}, fa_file, base, count),
                cores=1, mem='6GB', time=time,
            )
        )

    final = {}
    for out in outs:
        final.update(out.get())
    return final


###############################################################################
#                              Helper Functions                               #
###############################################################################


def seq(sequence):
    """Convert Bio.Seq object to string."""
    return str(sequence.seq.upper())


def get_fasta_file(directory, name):
    """Look in directory for name.fa or name.fa.gz and return path."""
    fa_file = os.path.join(directory, name + '.fa')
    gz_file = fa_file + '.gz'
    if os.path.isfile(fa_file):
        genome_file = fa_file
    elif os.path.isfile(gz_file):
        genome_file = fa_file
    else:
        raise FileNotFoundError(
            'No {f}.fa or {f}.fa.gz file found in {d}'.format(
                f=name, d=directory
            )
        )
    return genome_file


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
#                                Run On Files                                 #
###############################################################################


def parse_location_file(infile, base=None):
    """Get a compatible dictionary from an input file.

    Args:
        infile (str): Path to a bed, vcf, or tsv. If tsv should be chrom\\tpos.
                      Filetype detected by extension. Gzipped/B2zipped OK.
        base (int):   Force base of file, if not set, bed/tsv assumed base 0,
                      vcf assumed base-1

    Returns:
        dict: A dict of {chrom->pos}
    """
    if not isinstance(base, int):
        base = 1 if 'vcf' in infile.split('.') else 0
    out = {}
    for chrom, pos in tsv_bed_vcf(infile, base):
        if chrom not in out:
            out[chrom] = []
        out[chrom].append(pos)
    return out


def tsv_bed_vcf(infile, base=0):
    """Interator for generic tsv, yields column1, column2 for every line.

    column1 is assumed to be string, column2 is converted to int and base is
    subtracted from it.
    """
    with open_zipped(infile) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            f = line.rstrip().split('\t')
            yield f[0], int(f[1])-base


###############################################################################
#                               Run as a Script                               #
###############################################################################

DESC = """\
Get a region for every position in a given file.

Will write out one line for every line in your file, as::
    chrom\\tposition\\tsequence

Parallelizes on the cluster if there is more than one chromosome.
"""

def get_parser():
    """Returns an argument parser."""
    parser  = argparse.ArgumentParser(
        description=DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument(
        'location_file',
        help="File of locations, bed, vcf, or just chrom\\tposition fine."
    )

    # Optional flags
    parser.add_argument('-d', '--distance', default=7,
                        help="Distance up and downstream of position to get")
    parser.add_argument('-g', '--genome', default=hg19,
                        help="Genome fasta/directory, defaults to hg19.")
    parser.add_argument('-o', '--out', default=sys.stdout,
                        help="File to write to, default STDOUT.")

    return parser


def main(argv=None):
    """Run using files, for running as a scipt."""
    if not argv:
        argv = sys.argv[1:]

    # Get arguments
    parser = get_parser()
    args   = parser.parse_args(argv)

    locations = parse_location_file(args.location_file)

    if len(locations) == 1:
        results = get_regions(locations, args.genome, 0, args.distance)
    else:
        results = get_regions_parallel(locations, args.genome, 0,
                                       args.distance)

    s = '{}\t{}\t{}\n'
    with open_zipped(args.out) as fout:
        for chrom, positions in locations.items():
            for position, sequence in positions.items():
                fout.write(s.format(chrom, position, sequence))

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
