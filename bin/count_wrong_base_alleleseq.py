#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loop through AlleleSeq .cnt files and pull out incorrect base counts from read.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       CREATED: 2016-18-07 16:06
 Last modified: 2016-06-07 21:15

============================================================================
"""
import os
import sys
import pickle
import argparse
import gzip
import bz2
import pandas as pd


###############################################################################
#                              Primary Function                               #
###############################################################################


def count_read_positions(bed_file, cnt_files, outfile=sys.stdout, pandas=None):
    """Count ref/alt/other/N for every snp and write to outfile.

    :bed_file:  Bed file of SNPs
    :cnt_files: List of .cnt files
    :outfile:   The file to write to.
    :returns:   None.

    """
    snps = parse_bed(bed_file, base=1)  # AlleleSeq is 1 based

    data = {'snp_count': [], 'skipped': [], 'ref': [], 'alt': [], 'other': [],
            'ns': [], 'total': [], 'chrom': [], 'tissue': []}

    # Loop through count files and tally
    for cnt_file in cnt_files:
        with open(cnt_file, 'rb') as fin:
            cnts = pickle.load(fin)
        for chrom, csnps in cnts.items():
            # Initialize counters
            ref       = 0
            alt       = 0
            other     = 0
            ns        = 0
            snp_count = 0
            skipped   = 0
            for pos, counts in csnps.items():
                snp_count += 1
                snp = snps[chrom][pos] if pos in snps[chrom] else None
                if not snp:
                    skipped += 1
                    continue
                ref   += counts[snp.ref.lower()]
                alt   += counts[snp.alt.lower()]
                ns    += counts['n']
                bases = set(counts.keys())
                for base in bases.difference([snp.ref.lower(), snp.alt.lower(), 'n']):
                    other += counts[base]
            data['snp_count'].append(snp_count)
            data['skipped'].append(skipped)
            data['ref'].append(ref)
            data['alt'].append(alt)
            data['other'].append(other)
            data['ns'].append(ns)
            data['total'].append(ref+alt+other+ns)
            data['other_percent'].append(other/(ref+alt+other+ns))
            data['chrom'].append(chrom)
            data['tissue'].append(os.path.basename(cnt_file).split('.')[0])
            assert isinstance(data, dict)

    # Make dataframe
    df = pd.DataFrame.from_dict(data)
    df = df[['tissue', 'chrom', 'snp_count', 'skipped', 'ref', 'alt', 'ns',
             'other', 'other_percent', 'total']]

    # Save pandas
    if pandas:
        df.to_pickle(pandas)

    # Print output
    df.to_csv(outfile, sep='\t')


###############################################################################
#                              Bed File Parsing                               #
###############################################################################


class SNP(object):
    """ A simple object to store the ref and alt alleles of a SNP """

    def __init__(self, ref, alt):
        self.ref = ref
        self.alt = alt

    def __repr__(self):
        return "Ref:{0}\tAlt:{1}".format(self.ref, self.alt)


class Chromosome(object):
    """ A chromosome container, holds a dictionary of SNP objects,
        which correspond to every SNP on that chromosome.
        Also holds lists of ref, alt, and no_match SNPs, which are
        created by get_lists(), which takes a SeqIO chromosome that
        must be the same as this chromosome.
    """

    def add_snp(self, position, snp):
        """ Add a SNP object to the dictionary of snps """
        self.snps[position] = snp

    def __init__(self, name):
        self.snps        = {}
        self.name        = name
        self.number      = name[3:] if name.startswith('chr') else name

    def __repr__(self):
        out_string = "Chromosome: {0} ({1})\n".format(self.name, self.number)
        out_string = out_string + '{0:>30}: {1}\n\n'.format(
            "Total SNP Count", len(self.snps))
        return out_string

    def __len__(self):
        return len(self.snps)

    def __getitem__(self, key):
        return self.snps[key]

    def __contains__(self, key):
        return key in self.snps


def parse_bed(bed_file, base=0):
    """ Return a dictionary of Chromosome objects from a bed file
        File can be plain, gzipped, or bz2zipped
        'base' is added to the position
    """
    try:
        assert isinstance(base, int)
    except AssertionError:
        raise AssertionError('base must be an integer, is {}'.format(
            type(base)))

    chromosomes = {}

    with open_zipped(bed_file) as infile:
        for line in infile:
            if line.startswith('#'):
                continue
            f = line.rstrip().split('\t')
            if f[0] not in chromosomes:
                chromosomes[f[0]] = Chromosome(f[0])
            ref, alt = f[3].split('|')
            chromosomes[f[0]].add_snp(int(f[1])+base, SNP(ref, alt))

    return chromosomes

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
#                            Command Line Parsing                             #
###############################################################################


def main(argv=None):
    """Parse command line options."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('bed_file',
                        help="Bed file of SNP positons with ref/alt")
    parser.add_argument('cnt_files', nargs='+',
                        help="Alleleseq cnt files")

    parser.add_argument('-o', '--outfile', default=sys.stdout,
                        help="Output file, Default STDOUT")
    parser.add_argument('-p', '--pandas',
                        help="Output file, Default STDOUT")

    args = parser.parse_args(argv)

    count_read_positions(args.bed_file, args.cnt_files, args.outfile,
                         args.pandas)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
