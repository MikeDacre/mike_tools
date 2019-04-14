#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize effect sizes for ATACseq

       Created: 2016-47-26 13:09
 Last modified: 2016-09-26 17:45
"""
import os
import sys
import argparse
from time import sleep

from multiprocessing import Pool

import numpy as np
import pandas as pd

import cluster


###############################################################################
#                           Normalization functions                           #
###############################################################################


def norm_both(df):
    """Normalize an effect size.

    Intended to use with a pandas data frame.

    Algorithm: if below line: pre-post/pre; if below line: pre-post/1-pre

    :df:      Pandas dataframe with the columns 'pre' and 'post'
    :returns: Pandas dataframe with an additional 'norm_effect' column

    """
    df['norm_effect'] = \
        ((df.pre > df.post) * (df.pre - df.post)/df.pre) + \
        ((df.pre <= df.post) * (df.pre - df.post)/(1 - df.pre))
    return df


def norm_max(df):
    """Normalize an effect size.

    Intended to use with a pandas data frame.

    Algorithm: pre-post/max(pre, 1-pre)

    :df:      Pandas dataframe with the columns 'pre' and 'post'
    :returns: Pandas dataframe with an additional 'norm_effect' column

    """
    df['norm_effect'] = (df.pre - df.post)/df.pre.clip(lower=1-df.pre)
    return df


def norm_min(df):
    """Normalize an effect size.

    Intended to use with a pandas data frame.

    Algorithm: pre-post/min(pre, 1-pre)

    :df:      Pandas dataframe with the columns 'pre' and 'post'
    :returns: Pandas dataframe with an additional 'norm_effect' column

    """
    df['norm_effect'] = (df.pre - df.post)/df.pre.clip(upper=1-df.pre)
    return df


def stderr(df):
    """Calculate the stderr and variance on a dataframe.

    First calculate a variance as sqrt(prevar + postvar)
    Then calculate a z-score as pre-post/var
    Next, calculate a stddev as abs(norm_effect/z)
    Finally, create a beta variance as sqrt(stddev)

    :df:      Pandas dataframe with the columns 'pre' 'post' 'prevar' 'postvar'
              'norm_effect'
    :returns: Same dataframe with additional columns: 'z' 'vari' 'stddev'
              'bvari' 'preminuspost'

    """
    df['vari']   = np.sqrt(df.prevar + df.postvar)
    df['z']      = (df.pre-df.post)/df.vari
    df['stddev'] = (df.norm_effect/df.z).abs()
    df['bvari']  = np.sqrt(df.stddev)
    return df


def rename_snp(x, dct):
    """For use with Series.apply."""
    try:
        return dct[x]
    except KeyError:
        return np.nan


###############################################################################
#                               Main functions                                #
###############################################################################


def normalize_files(norm_by, prefix, rename_file, files):
    """Submit a normalization job for every file to the cluster.

    :norm_by:     max, min, or both
    :prefix:      What name to append to the output file names
    :rename_file: A file with CHR.SNP\\tNew Name
    :files:       A list of files to submit

    """
    pool = Pool(16)
    jobs = []
    for fl in files:
        jobs.append(pool.apply_async(normalize_file,
                                     (norm_by, prefix, rename_file, fl)))
    for job in jobs:
        job.get()


def normalize_file(norm_by, prefix, rename_file, infile):
    """Normalize the effect sizes by either max, min, or both

    :norm_by:     max, min, or both
    :prefix:      What name to append to the output file names
    :rename_file: A file with CHR.SNP\\tNew Name
    :infile:      A tab delimited file with the following columns::
        ['chr', 'position', 'ref', 'depth', 'post',
        'pre', 'pval', 'prevar', 'postvar']

    Columns must be in exactly that order, column names are ignored.

    MAF is limited to between 0.02 and 0.98.

    Output is chr.name\\tpop\\teffect\\tstderr

    chr.name is renamed by the index in rename_file.

    """
    # Read dataframe
    df = pd.read_csv(infile, sep='\t')

    # Rename columns
    df.columns = ['chrom', 'position', 'ref', 'depth', 'post',
                  'pre', 'pval', 'prevar', 'postvar']
    df.sort_values(['chrom', 'position'], inplace=True)

    # Filter MAF
    df = df[df.pre < 0.98]
    df = df[df.pre > 0.02]

    # Get population from name
    path, name = os.path.split(infile)
    pop = name.split('.')[0]
    df['pop'] = pop

    # Create name
    df['snp'] = df.chrom + '.' + df.position.apply(str)

    # Get renamed name
    with open(rename_file) as fin:
        rename_dict = {}
        for line in fin:
            orig, new = line.rstrip().split('\t')
            rename_dict[orig.strip()] = new.strip()
    df['name'] = df.snp.apply(rename_snp, args=(rename_dict,))
    del(rename_dict)

    # Normalize effect
    if norm_by == 'max':
        norm_func = norm_max
    elif norm_by == 'min':
        norm_func = norm_min
    elif norm_by == 'both':
        norm_func = norm_both
    else:
        raise Exception("Norm by value not recognized {}".format(norm_by))

    df = norm_func(df)

    # Calculate z-score and errors
    df = stderr(df)

    # Reorganize columns
    df = df[['chrom', 'position', 'snp', 'name', 'pop', 'ref', 'depth', 'pre',
             'post', 'prevar', 'postvar', 'z', 'vari', 'pval', 'norm_effect',
             'stddev', 'bvari']]

    # Print output
    g = df[df.stddev.notnull()]
    beta_df = g[['name', 'pop', 'norm_effect', 'bvari']]
    beta_df = beta_df[beta_df.name.notnull()]
    df.to_pickle(os.path.join(path, prefix + '.' + name + '.pandas'))
    beta_df.to_csv(os.path.join(path, prefix + '.' + name + '.betas.txt'),
                   index=False, sep='\t')


def main(argv=None):
    """Parse command line args. """
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('rename_file',
                        help="File to use to name by peak. (SNP\\tname)")
    parser.add_argument('files', nargs='+',
                        help="Input files from Ashlye's pipleine")

    # Optional flags
    parser.add_argument('-n', '--normalize-by', choices={'max', 'min', 'both'},
                        default='both', help="Which factor to normalize by")
    parser.add_argument('-p', '--prefix', default='normalized',
                        help="Prefix to use on output files")

    args = parser.parse_args(argv)

    normalize_files(args.normalize_by, args.prefix, args.rename_file,
                    args.files)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
