#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check all reads in a sam file and check provided SNPs to see if is ref/alt.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       CREATED: 2016-11-08 14:06
 Last modified: 2016-06-23 13:28

============================================================================
"""
import os
import sys
import pickle
import argparse
import pandas as pd
import pysam
#  from tqdm import tqdm as pb


def parse_bed(bedfile, overwrite=False):
    """Return a dictionary of dataframes of all SNPs in the bed file.

    Expects a bed file with ref|alt instead of name.

    Dictionary:: {chr->df(position, ref, alt)}
    """

    cached_snps = bedfile + '.pickle'
    if os.path.exists(cached_snps) and not overwrite:
        return pd.read_pickle(cached_snps)

    df = pd.read_csv(bedfile, sep='\t', usecols=[0, 1, 3],
                     names=['chrom', 'start', 'ref|alt'], engine='c',
                     header=None, index_col=[0,1])

    df = df.sort_index()

    dfs = {}
    for chrom in df.index.levels[0]:
        dfs[chrom] = df.loc[chrom]['ref|alt']

    with open(cached_snps, 'wb') as fout:
        pickle.dump(dfs, fout)

    return dfs

flipper = {'C': 'G', 'A': 'T', 'G': 'C', 'T': 'A', 'N': 'N',
           'c': 'g', 'a': 't', 'g': 'c', 't': 'a', 'n': 'n'}

def read_samfile(infile, flip=True):
    """Simple iterator to yield chr,pos,end,seq for every sam line.

    NOTE: Strips chromosome label to remove anything after '_',
          e.g. chr1_paternal becomes chr1.

    :flip: Convert the whole strand to the reverse complement
    """
    with open(infile) as fin:
        for line in fin:
            f = line.rstrip().split('\t')
            chrom = f[2].split('_')[0]  # Get rid of _paternal/maternal
            pos   = int(f[3])-1 ### Convert to base-0 ###
            seq   = f[9]
            end   = pos+len(seq)-1
            flag  = int(f[1])
            is_reverse = flag&16 == 16
            if flip and not is_reverse:
                seq = ''.join([flipper[s] for s in seq])
            yield chrom, pos, end, seq, is_reverse


def read_bamfile(infile, flip=True):
    """Simple iterator to yield chr,pos,end,seq for every bam line.

    NOTE: Strips chromosome label to remove anything after '_',
          e.g. chr1_paternal becomes chr1.

    :flip: Convert the whole strand to the reverse complement
    """
    afile = pysam.AlignmentFile(infile)
    for r in afile:
        chrom = r.reference_name.split('_')[0]  # Get rid of _paternal/maternal
        pos   = r.pos-r.qstart
        seq   = r.seq
        end   = pos+len(seq)-1
        if flip and r.is_reverse:
            seq = ''.join([flipper[s] for s in seq])
        yield chrom, pos, end, seq, r.is_reverse


def read_bam_sam(infile, flip=True):
    """Iterator to yield chr,pos,end,seq for either bam or sam."""
    if infile.endswith('bam'):
        return read_bamfile(infile, flip)
    elif infile.endswith('sam'):
        return read_samfile(infile, flip)
    else:
        raise Exception('File must end with bam or sam')


def check_reads(snps, sam_files, outfile=sys.stdout, pandas=None,
                logfile=sys.stderr, flip=True):
    """Check all reads in all sam files and compile statistics on ref/alt.

    Will create a pandas dataframe of summary statistics for every SNP
    position as well dataframes for the position +/- 1.

    Summary stats include: ref_count, alt_count, n_count, other_count

    :snps:      A SNP dictionary returned from parse_bed
    :sam_files: A list of paths to sam files
    :outfile:   A file to write a summary table to (default STDOUT)
    :pandas:    A file to pickle a dictionary of pandas dataframes
    :logfile:   A file to write simple stats (default STDERR)
    :flip:      Convert the whole strand to the reverse complement
    :returns:   A dictionary of dataframes

    """
    fileinfo  = {}
    for samfile in sam_files:
        #  name = '.'.join(os.path.basename(samfile).split('.')[:-1])
        name = samfile.split('/')[0]
        fileinfo[name] = {'ref': 0, 'alt': 0, 'other': 0, 'ns': 0, 'reads': 0,
                          'with_snps': 0, 'snps': 0, 'base': 0}
        fileinfo[name + '_1'] = {'ref': 0, 'alt': 0, 'other': 0, 'ns': 0,
                                 'reads': 0, 'with_snps': 0, 'snps': 0,
                                 'base': 0}
        fileinfo[name + '_m1'] = {'ref': 0, 'alt': 0, 'other': 0, 'ns': 0,
                                  'reads': 0, 'with_snps': 0, 'snps': 0,
                                  'base': 0}
        logfile.write('Working on {}\n'.format(name))
        #  filelen = int(check_output(['wc', '-l', samfile]).decode()
                      #  .split(' ')[0])
        #  for chr, start, end, seq in pb(read_samfile(samfile), unit='reads',
        #                                 total=filelen):
        for chrom, start, end, seq, _ in read_bam_sam(samfile, flip):
            for nm, bs in [(name, 0), (name + '_1', 1), (name + '_m1', -1)]:
                # To get info on alternate bases, I subtract bs from the start
                # and end values. This means '_1' ends up being base1, and
                # '_m1' ends up being base-1. I shift the whole read in the
                # opposite direction, with the effect being that the SNP
                # encoded in the bed file is has the correct adjusted base
                # value.
                astart = start - bs
                aend   = end - bs
                fileinfo[nm]['reads'] += 1
                try:
                    snplist = snps[chrom].loc[astart:aend]
                except KeyError:
                    pass
                if snplist.empty:
                    continue
                fileinfo[nm]['with_snps'] += 1
                # Convert the Series to a dictionary of pos->'ref|alt'
                snplist = snplist.to_dict()
                fileinfo[nm]['snps'] += len(snplist)
                for pos, info in snplist.items():
                    base = seq[pos-astart].lower()
                    info = info.lower()
                    if base == info[0]:
                        fileinfo[nm]['ref'] += 1
                    elif base == info[2]:
                        fileinfo[nm]['alt'] += 1
                    elif base == 'n':
                        fileinfo[nm]['ns'] += 1
                    else:
                        fileinfo[nm]['other'] += 1
        sys.stderr.write('{} done: {}\n'.format(samfile, fileinfo))

    df = pd.DataFrame.from_dict(fileinfo, orient='index')

    if pandas:
        df.to_pickle(pandas)

    df.to_csv(outfile, sep='\t')


def main(argv=None):
    """Command line arguments"""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('snp_file',
                        help="Bed file of SNPs to check")
    parser.add_argument('sam_files', nargs='+',
                        help="Sam files to check")

    # Optional Arguments
    parser.add_argument('-f', '--flip', action='store_true',
                        help="If read is reverse strand, flip to reference "
                             "before proceeding")

    # Optional Files
    optfiles = parser.add_argument_group('Optional Files')
    optfiles.add_argument('-p', '--pandas',
                          help="Pandas output file")
    optfiles.add_argument('-o', '--outfile', default=sys.stdout,
                          help="Output file, Default STDOUT")
    optfiles.add_argument('-l', '--logfile', default=sys.stderr,
                          help="Log File, Default STDERR (append mode)")

    args = parser.parse_args(argv)

    snps = parse_bed(args.snp_file)

    check_reads(snps, args.sam_files, args.outfile, args.pandas, args.logfile)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
