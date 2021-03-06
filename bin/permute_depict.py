#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run DEPICT on two datasets with a permutation and Wilcoxin test.

===============================================================================

AUTHOR:       Michael D Dacre, mike.dacre@gmail.com
ORGANIZATION: Stanford University
LICENSE:      MIT License, property of Stanford, use as you wish
VERSION:      0.1
CREATED:      2017-45-04 09:04

===============================================================================

[DEPICT](https://data.broadinstitute.org/mpg/depict) is an excellent tool from
the Broad institute that allows rapid gene and tissue enrichment analysis from
only a list of rsids.

This script allows you to run depict on two datasets and compare the results,
to determine significance, the datasets are permuted (randomly shuffled from
one list to another) and DEPICT is run again and the two random samples. This
is done repeatedly (default 100 times) and the results are compared to actual
results using a Wilcoxin test.

USAGE:
    Import as a module or run as a script.

REQUIREMENTS:
    java for running DEPICT
    DEPICT to be downloaded to a script accessible PATH (must be available to
    all cluster nodes)

    [fyrd](github.com/MikeDacre/fyrd): run jobs on a cluster
    numpy, scipy, matplotlib, and pandas: data analysis
"""
import os as _os
from os.path import join as _pth

import sys as _sys

import bz2 as _bz2
import gzip as _gzip
import argparse as _argparse

from subprocess import check_call as _call
import multiprocessing as _mp

from time import sleep as _sleep

import dill as _pickle

from tqdm import tqdm, tqdm_notebook

import numpy as np
from scipy import stats as sts
import pandas as pd

import fyrd as _fyrd

try:
    if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
        pb = tqdm_notebook
    else:
        pb = tqdm
except NameError:
    pb = tqdm

###############################################################################
#                  Set Defaults For Script - Modify At Will                   #
###############################################################################

DATA   = '/godot/atacseq/depict'         # Root directory for running
DEPICT = '/godot/atacseq/depict/depict'  # Where the DEPICT progam is

# Depict Runtime Flags, generally you should leave these alone
FLAG_LOCI     = 1  # Construct loci based on your associated SNPs
FLAG_GENES    = 1  # Prioritize Genes
FLAG_GENESETS = 1  # Conduct reconstituted gene set enrichment analysis
FLAG_TISSUES  = 1  # Conduct tissue/cell type enrichment analysis
PARAM_NCORES  = 4  # Number of cores to use *PER PROCESS* for DEPICT

###############################################################################
#                          Do Not Modify Below Here                           #
###############################################################################

# --------------------------------------------------------------------------- #


###############################################################################
#                              Master Functions                               #
###############################################################################


def analyze_depict(sample_1, sample_2, prefix, cores=None, perms=100,
                   run_path=None, depict_path=DEPICT, **fyrd_args):
    """Run depict, run permutations, compare results.

    This function uses fyrd to submit cluster jobs, jobs will request 2*cores
    to run, and 12G of memory.

    For 100 permutations, this function takes about 3.5 hours to run.

    Args:
        sample_1 (str):    File name or path to file with rsids for sample 1
        sample_2 (str):    File name or path to file with rsids for sample 2
        prefix (str):      Name for the output directory, input file names will
                           be used to set output files in this directory.
        cores (int):       Number of cores to use *PER PROCESS* for DEPICT,
                           defaults to 1/2 of available cores on the machine,
                           meaning all cores will be used for run (1/2 each).
        perms (int):       Number of permutations.
        run_path (str):    Root directory to run in, defaults to current dir
        depict_path (str): Path to the DEPICT package, default set in file.
        fyrd_args (dict):  Fyrd keyword arguments, not required.

    Outputs:
        <prefix>/<sample_name>.geneprioritization.txt
        <prefix>/<sample_name>.loci.txt
        <prefix>/<sample_name>.tissueenrichment.txt
        <prefix>/<sample_name>.genesetenrichment.txt
        <prefix>/<sample_name>.log

    Returns:
        DataFrame, DataFrame: All gene/tissue permutation outputs in two data
                              frames with a permutation column added to
                              distinguish data.
    """
    print('Submitting main DEPICT job to cluster')
    if not cores:
        cores = PARAM_NCORES
    run_path = _os.path.abspath(run_path)
    if not _os.path.isdir(run_path):
        _os.mkdir(run_path)

    startdir = _os.path.abspath(_os.path.curdir)

    try:
        job = _fyrd.Job(run_parse_depict,
                        (sample_1, sample_2, prefix, cores, run_path, depict_path),
                        name = 'main_DEPICT',
                        cores = cores*2,
                        mem = '12GB',
                        scriptpath = run_path,
                        outpath = run_path,
                        runpath = run_path,
                        **fyrd_args)
        job.submit()
        print('Job submitted.')

        print('Run permutations')
        pgenes, ptissues = permute_depict(sample_1, sample_2, prefix, cores, perms,
                                        run_path, depict_path, **fyrd_args)
        pgenes.to_pickle('pgenes.bak')
        ptissues.to_pickle('ptissues.bak')

        print('Permutations complete, getting main output')

        genes, tissues = job.get()

        genes.to_pickle('genes.bak')
        tissues.to_pickle('tissues.bak')

        print('Main job completed successfully, DataFrames saved.')

        data = {'genes': genes, 'tissues': tissues,
                'pgenes': pgenes, 'ptissues': ptissues}

        with open(prefix + '_completed_dfs.pickle', 'wb') as fout:
            _pickle.dump(data, fout)

        data = examine_data(**data)
    finally:
        _os.chdir(startdir)

    return data


def examine_data(genes, tissues, pgenes, ptissues,
                 sample_1_name='sample_1', sample_2_name='sample_2'):
    """Examine significance of gene and tissue data using permutations.

    Uses four dataframes: the gene and tissue data from run_parse_depict() and
    the permuted versions of the same from permute_depict().

    Returns:
        dict: {'stats': data, (dict)
               'genes': genes, (DF)
               'tissues': tissues, (DF)
               'perm_genes': pgenes, (DF)
               'perm_tissues': ptissues, (DF)
               'perm_genes_pivot': pg, (DF)
               'perm_tissues_pivot': pt (DF)}
    """
    data = {'genes': {}, 'tissues': {}}

    print('Comparing the gene set pvalues to the permuted version.')
    gm = sts.mannwhitneyu(genes.pldiff, pgenes.pldiff)
    data['genes']['mannwhitney'] = gm
    data['genes']['mannw_p'] = gm.pvalue

    # Pivot table based on absolute difference of the log10 of sample1/2
    pg = pgenes.pivot_table(
        index='gene_set_ID', columns='perm', values='pldiff'
    )
    pg = add_data_to_perm_pivot(pg)

    genes = pd.merge(genes, pg[['gene_set_ID', 'max', 'median', 'mean',
                                '95th_percentile']],
                     on='gene_set_ID')
    genes = genes.rename(columns={'max': 'max_log_p_diff_perms',
                                  'median': 'median_log_p_diff_perms',
                                  'mean': 'mean_log_p_diff_perms',
                                  '95th_percentile': 'p95_log_p_diff_perms'})
    genes['beats_perm'] = genes['pldiff'] > genes['max_log_p_diff_perms']
    genes['beats_perm95'] = genes['pldiff'] > genes['p95_log_p_diff_perms']

    print('Comparing the tissue set pvalues to the permuted version.')
    tm = sts.mannwhitneyu(tissues.pldiff, ptissues.pldiff)
    data['tissues']['mannwhitney'] = tm
    data['tissues']['mannw_p'] = tm.pvalue

    # Pivot table based on absolute difference of the log10 of sample1/2
    pt = ptissues.pivot_table(
        index='MeSH_term', columns='perm', values='pldiff'
    )
    pt = add_data_to_perm_pivot(pt)

    tissues = pd.merge(tissues, pt[['MeSH_term', 'max', 'median', 'mean',
                                    '95th_percentile']],
                       on='MeSH_term')
    tissues = tissues.rename(
        columns={'max': 'max_log_p_diff_perms',
                 'median': 'median_log_p_diff_perms',
                 'mean': 'mean_log_p_diff_perms',
                 '95th_percentile': 'p95_log_p_diff_perms'}
    )
    tissues['beats_perm'] = tissues['pldiff'] > tissues['max_log_p_diff_perms']
    tissues['beats_perm95'] = tissues['pldiff'] > tissues['p95_log_p_diff_perms']

    return {
        'stats': data,
        'genes': genes,
        'tissues': tissues,
        'perm_genes': pgenes,
        'perm_tissues': ptissues,
        'perm_genes_pivot': pg,
        'perm_tissues_pivot': pt,
    }


def add_data_to_perm_pivot(df):
    """Add some summary statistics to a pivot table of permuted data."""
    # Sort permutations
    df = df[[str(j) for j in sorted([int(i) for i in df.columns])]].copy()
    # Basic summary stats
    dfmin    = df.apply(np.min, axis=1)
    dfmax    = df.apply(np.max, axis=1)
    dfmean   = df.apply(np.mean, axis=1)
    dfmedian = df.apply(np.median, axis=1)
    ptl95    = df.apply(np.percentile, args=(95,), axis=1)
    df['min'], df['max'], df['mean'] = dfmin, dfmax, dfmean
    df['median'], df['95th_percentile'] = dfmedian, ptl95
    df.reset_index(drop=False, inplace=True)
    return df


def merge_data(genes, tissues, handle_perm=False,
               sample_1_name='sample_1', sample_2_name='sample_2'):
    """Merge gene and tissue dataframes and name samples

    Uses four dataframes: the gene and tissue data from run_parse_depict() and
    the permuted versions of the same from permute_depict().
    """
    # Names
    s1p = sample_1_name + '_p_value'
    s2p = sample_2_name + '_p_value'
    s1f = sample_1_name + '_FDR_lt_0.05'
    s2f = sample_2_name + '_FDR_lt_0.05'

    gs = ['gene_set_ID']
    ts = ['MeSH_term']
    if handle_perm:
        gs.append('perm')
        ts.append('perm')

    # Create intersected tables
    genes = pd.merge(
        genes[genes['sample'] == sample_1_name].rename(
            columns={'p_value': s1p, 'FDR_lt_0.05': s1f }
        ).drop('sample', axis=1),
        genes[genes['sample'] == sample_2_name].rename(
            columns={'p_value': s2p, 'FDR_lt_0.05': s2f}
        ).drop(['gene_set_description', 'sample'], axis=1),
        on=gs
    )
    tissues = pd.merge(
        tissues[tissues['sample'] == sample_1_name].rename(
            columns={'p_value': s1p, 'FDR_lt_0.05': s1f }
        ).drop('sample', axis=1),
        tissues[tissues['sample'] == sample_2_name].rename(
            columns={'p_value': s2p, 'FDR_lt_0.05': s2f}
        ).drop(
            ['MeSH_first_level_term', 'MeSH_second_level_term',
             'sample'], axis=1
        ),
        on=ts
    )

    for df in [genes, tissues]:
        for s in [s1p, s2p]:
            df['log10_' + s] = np.log10(df[s])
        df['pdiff']   = np.abs(df[s1p] - df[s2p])
        df['pldiff']  = np.abs(df['log10_' + s1p] - df['log10_' + s2p])

    return genes, tissues


def run_parse_depict(sample_1, sample_2, prefix, cores=None,
                     run_path=None, depict_path=DEPICT):
    """Run run_depict() once and parse results.

    Parallelization at this step will be local only.

    Args:
        sample_1 (str):    File name or path to file with rsids for sample 1
        sample_2 (str):    File name or path to file with rsids for sample 2
        prefix (str):      Name for the output directory, input file names will
                           be used to set output files in this directory.
        cores (int):       Number of cores to use *PER PROCESS* for DEPICT,
                           defaults to 1/2 of available cores on the machine,
                           meaning all cores will be used for run (1/2 each).
        run_path (str):    Root directory to run in, defaults to current dir
        depict_path (str): Path to the DEPICT package, default set in file.

    Outputs:
        <prefix>/<sample_name>.geneprioritization.txt
        <prefix>/<sample_name>.loci.txt
        <prefix>/<sample_name>.tissueenrichment.txt
        <prefix>/<sample_name>.genesetenrichment.txt
        <prefix>/<sample_name>.log

    Returns:
        DataFrame, DataFrame: Gene and Tissue DataFrames with sample columns.
    """
    if not cores:
        cores = PARAM_NCORES
    main_datafiles = run_depict(sample_1, sample_2, prefix, cores, run_path,
                                depict_path)
    genes = []
    tissues = []
    for sample, files in main_datafiles.items():
        genes.append(parse_gene_file(files['gene'], {'sample': sample}))
        tissues.append(parse_tissue_file(files['tissue'], {'sample': sample}))

    # Increase sanity
    genes = pd.concat(genes).sort_values('sample').reset_index(drop=True)
    tissues = pd.concat(tissues).sort_values('sample').reset_index(drop=True)

    return merge_data(genes, tissues)


def parse_completed(sample_1_name, sample_2_name, path,
                    sample_1_prefix='sample_1', sample_2_prefix='sample_2'):
    """Return merged gene and tissue dataframes from a directory.

    Parameters
    ----------
    sample_1_name : str
        The prefix used in the file naming when DEPICT was run
    sample_2_name : str
    path : str
        The path to the directory to parse
    sample_1_prefix : str, optional
        A name to give the first sample
    sample_2_prefix : str, optional

    Returns
    -------
    genes : DataFrame
    tissues : DataFrame
    """
    s1 = _os.path.abspath(_os.path.join(path, sample_1_name))
    s2 = _os.path.abspath(_os.path.join(path, sample_2_name))
    s1g = s1 + '_genesetenrichment.txt'
    s2g = s2 + '_genesetenrichment.txt'
    s1t = s1 + '_tissueenrichment.txt'
    s2t = s2 + '_tissueenrichment.txt'
    genes = pd.concat(
        [
            parse_gene_file(s1g, {'sample': sample_1_prefix}),
            parse_gene_file(s2g, {'sample': sample_2_prefix})
        ]
    ).sort_values('sample').reset_index(drop=True)
    tissues = pd.concat(
        [
            parse_tissue_file(s1t, {'sample': sample_1_prefix}),
            parse_tissue_file(s2t, {'sample': sample_2_prefix})
        ]
    ).sort_values('sample').reset_index(drop=True)

    return merge_data(genes, tissues,
                      sample_1_name=sample_1_prefix,
                      sample_2_name=sample_2_prefix)


def parse_depict(datafiles):
    genes = []
    tissues = []
    for sample, files in datafiles.items():
        genes.append(parse_gene_file(files['gene'], {'sample': sample}))
        tissues.append(parse_tissue_file(files['tissue'], {'sample': sample}))

    # Increase sanity
    genes = pd.concat(genes).sort_values('sample').reset_index(drop=True)
    tissues = pd.concat(tissues).sort_values('sample').reset_index(drop=True)

    return merge_data(genes, tissues)


###############################################################################
#                             Handle Permutations                             #
###############################################################################


def permute_depict(sample_1, sample_2, prefix, cores=None, perms=100,
                   run_path=None, depict_path=DEPICT, **fyrd_args):
    """Run DEPICT permutations, load results.

    This function uses fyrd to submit cluster jobs, jobs will request 2*cores
    to run, and 12G of memory.

    Args:
        sample_1 (str):    File name or path to file with rsids for sample 1
        sample_2 (str):    File name or path to file with rsids for sample 2
        prefix (str):      Name for the output directory, input file names will
                           be used to set output files in this directory.
        cores (int):       Number of cores to use *PER PROCESS* for DEPICT,
                           defaults to 1/2 of available cores on the machine,
                           meaning all cores will be used for run (1/2 each).
        perms (int):       Number of permutations.
        run_path (str):    Root directory to run in, defaults to current dir
        depict_path (str): Path to the DEPICT package, default set in file.
        fyrd_args (dict):  Fyrd keyword arguments, not required.

    Outputs:
        <prefix>/<sample_name>.geneprioritization.txt
        <prefix>/<sample_name>.loci.txt
        <prefix>/<sample_name>.tissueenrichment.txt
        <prefix>/<sample_name>.genesetenrichment.txt
        <prefix>/<sample_name>.log

    Returns:
        DataFrame, DataFrame: All gene/tissue permutation outputs in two data
                              frames with a permutation column added to
                              distinguish data.
    """
    datafiles = run_depict_permutation(
        sample_1, sample_2, prefix, cores, perms, run_path, depict_path,
        **fyrd_args
    )
    return _permute_depict(datafiles)


def _permute_depict(datafiles):
    """Handle the datafile parsing for this function."""
    # Parse datafiles
    genes = []
    tissues = []
    for name, samples in datafiles.items():
        if isinstance(samples, tuple) or samples is None:
            continue
        perm = int(name.split('_')[-1])
        g = []
        t = []
        for sample, files in samples.items():
            g.append(parse_gene_file(files['gene'],
                                     {'perm': perm, 'sample': sample}))
            t.append(parse_tissue_file(files['tissue'],
                                       {'perm': perm, 'sample': sample}))
        g = pd.concat(g)
        t = pd.concat(t)
        g, t = merge_data(g, t, handle_perm=True)
        if len(g) > 0:
            genes.append(g)
        if len(t) > 0:
            tissues.append(t)

    # Merge data
    genes   = pd.concat(genes)
    tissues = pd.concat(tissues)

    # Increase sanity
    genes   = genes.sort_values(['perm', 'sample']).reset_index(drop=True)
    tissues = tissues.sort_values(['perm', 'sample']).reset_index(drop=True)

    return genes, tissues


def get_gene_perms(df, path, prefix, open_file, closed_file):
    """Use file and path info to get permutations and create combined data.

    Note: Not used in the primary workflow of this script, provided, like
    rescue_permutation as a recovery and analysis tool.

    Parameters
    ----------
    df : DataFrame
        A genes DataFrame that has already been fully merged and has pldiff
        info
    path : str
        Path to the permutations
    prefix : str
        Prefix used when running permutations
    open_file : str
        Prefix of the open file (excluding permutation info), e.g.:
            cad_open_is_risk_test.txt<_perm_#_.....txt> # last part excluded
    closed_file : str
        Same as open_file but for closed = risk

    Returns
    -------
    df : DataFrame
        Same as input DF but with permutation data added and indexing by
        gene set
    mwu : sts.mannwhitneyu
        A scipy.stats mwu object
    pgenes : DataFrame
        The merged permutation dataframe
    pvt : DataFrame
        The pivotted data
    """
    pgenes = []
    for fl in [_pth(_os.path.abspath(path), i) for i in _os.listdir(path) if i.startswith(prefix)]:
        if fl.endswith('files.txt'):
            continue
        perm = int(fl.split('_')[-1])
        try:
            d1 = pd.read_csv(_pth(fl, '{}_perm_{}_genesetenrichment.txt'.format(open_file, perm)), sep='\t')
            d2 = pd.read_csv(_pth(fl, '{}_perm_{}_genesetenrichment.txt'.format(closed_file, perm)), sep='\t')
        except:
            continue
        d1.columns = ['gene_set_ID', 'gene_set_description', 'p_value_open', 'FDR']
        d2.columns = ['gene_set_ID', 'gene_set_description', 'p_value_closed', 'FDR']
        d = pd.merge(d1, d2[['gene_set_ID', 'p_value_closed']], on='gene_set_ID')
        d['perm'] = perm
        pgenes.append(d)
    pgenes = pd.concat(pgenes)

    pgenes['log10_open'] = np.log10(pgenes.p_value_open)
    pgenes['log10_closed'] = np.log10(pgenes.p_value_closed)
    pgenes['pldiff'] = np.abs(pgenes.log10_closed - pgenes.log10_open)

    pvt = pgenes.pivot_table(index='gene_set_ID', columns='perm', values='pldiff')

    pvt['max_pldiff'] = pvt.apply(np.max, axis=1)

    df.set_index('gene_set_ID', drop=False, inplace=True)

    df['max_pldiff'] = pvt.max_pldiff

    df['beats_perm'] = pgenes.pldiff > pgenes.max_pldiff

    mwu = sts.mannwhitneyu(df.pldiff, pgenes.pldiff)

    return df, mwu, pgenes, pvt


###############################################################################
#                             DEPICT File Parsing                             #
###############################################################################


def parse_gene_file(infile, add_column=None):
    """Parse gene file from DEPICT, return DataFrame.

    Args:
        infile (str):       File to parse
        add_columns (dict): Add these columns to DF, scalar only

    Returns:
        DataFrame
    """
    df = pd.read_csv(infile, sep='\t')
    df.columns = ['gene_set_ID', 'gene_set_description',
                  'p_value', 'FDR_lt_0.05']
    if add_column:
        for name, data in add_column.items():
            df[name] = data
    return df


def parse_tissue_file(infile, add_columns=None):
    """Parse tissue file from DEPICT, return DataFrame.

    Args:
        infile (str):       File to parse
        add_columns (dict): Add these columns to DF, scalar only

    Returns:
        DataFrame
    """
    df = pd.read_csv(infile, sep='\t')
    df.columns = ['MeSH_term', 'MeSH_first_level_term',
                  'MeSH_second_level_term', 'p_value', 'FDR_lt_0.05']
    if add_columns:
        for name, data in add_columns.items():
            df[name] = data
    return df


###############################################################################
#                DEPICT Management Functions (Job Submission)                 #
###############################################################################


def run_depict(sample_1, sample_2, prefix, cores=None,
               run_path=None, depict_path=DEPICT):
    """Run DEPICT twice, once on each sample, DEPICT will be run in parallel.

    Parallelization at this step will be local only.

    Args:
        sample_1 (str):    File name or path to file with rsids for sample 1
        sample_2 (str):    File name or path to file with rsids for sample 2
        prefix (str):      Name for the output directory, input file names will
                           be used to set output files in this directory.
        cores (int):       Number of cores to use *PER PROCESS* for DEPICT,
                           defaults to 1/2 of available cores on the machine,
                           meaning all cores will be used for run (1/2 each).
        run_path (str):    Root directory to run in, defaults to current dir
        depict_path (str): Path to the DEPICT package, default set in file.

    Outputs:
        <prefix>/<sample_name>.geneprioritization.txt
        <prefix>/<sample_name>.loci.txt
        <prefix>/<sample_name>.tissueenrichment.txt
        <prefix>/<sample_name>.genesetenrichment.txt
        <prefix>/<sample_name>.log

    Returns:
        dict: Dictionary of relevant files. Raises Exception on error.
    """
    FLAG_LOCI     = 1  # Construct loci based on your associated SNPs
    FLAG_GENES    = 1  # Prioritize Genes
    FLAG_GENESETS = 1  # Conduct reconstituted gene set enrichment analysis
    FLAG_TISSUES  = 1  # Conduct tissue/cell type enrichment analysis
    PARAM_NCORES  = 4  # Number of cores to use *PER PROCESS* for DEPICT

    if not cores:
        cores = PARAM_NCORES
    # Set dirs
    startdir = _os.path.abspath(_os.path.curdir)
    run_path = _os.path.abspath(run_path if run_path else startdir)
    print(run_path)
    if not _os.path.isdir(run_path):
        _os.mkdir(run_path)
    # Get DEPICT dir
    if not _os.path.isdir(depict_path):
        depict_path = DEPICT
    depict_path = _os.path.abspath(depict_path)
    print(depict_path)

    #  check_depict(depict_path)

    # Check sample files
    infiles = {
        'sample_1': _os.path.abspath(sample_1),
        'sample_2': _os.path.abspath(sample_2),
    }
    for sample in infiles.values():
        if not _os.path.isfile(sample):
            raise FileNotFoundError('{} does not exist'.format(sample))
        with open(sample) as fin:
            assert fin.readline().strip().startswith('rs')

    try:
        # Change directory
        _os.chdir(run_path)

        # Create prefix, prefix_long is in run_path, prefix is in depict_path
        prefix = _os.path.basename(prefix)
        prefix_long = _os.path.abspath(prefix)
        if not _os.path.isdir(prefix_long):
            _os.makedirs(prefix_long)

        # Get names
        names = {
            'sample_1': '.'.join(_os.path.basename(sample_1).split('.')[:-1]),
            'sample_2': '.'.join(_os.path.basename(sample_2).split('.')[:-1]),
        }
        prefixes = {
            'sample_1_long': _pth(prefix_long, names['sample_1']),
            'sample_2_long': _pth(prefix_long, names['sample_2']),
            'sample_1': _pth(prefix, names['sample_1']),
            'sample_2': _pth(prefix, names['sample_2']),
        }

        # Set cores
        if not cores:
            cores = int(_mp.cpu_count()/2)

        # Change directory
        _os.chdir(depict_path)
        print(depict_path)
        if not _os.path.isdir(prefix):
            _os.makedirs(prefix)

        # Create script templates
        loci_script = (
            "java -Xms512m -Xmx4000m -jar "
            "{depict}/LocusGenerator/LocusGenerator.jar "
            "{depict}/LocusGenerator/config.xml {infile} "
            "{prefix} > {prefix}.log 2>&1"
        )
        gene_script = (
            "java -Xms512m -Xmx16000m -jar {depict}/Depict/Depict.jar "
            "{outname} {flag_genes} {flag_genesets} 0 {cores} {outdir} "
            ">> {prefix}.log 2>&1"
        )
        tissue_script = (
            "java -Xms512m -Xmx16000m -jar {depict}/Depict/Depict.jar "
            "{outname} 0 1 1 {cores} {outdir} >> {prefix}.log 2>&1"
        )

        # Run jobs
        if FLAG_LOCI:
            print('Running loci building job..')
            loci_jobs = {}
            # Create jobs
            for sample in ['sample_1', 'sample_2']:
                loci_jobs[sample] = _mp.Process(
                    target = _call,
                    args   = (
                        loci_script.format(
                            depict=depict_path, infile=infiles[sample],
                            prefix=prefixes[sample]
                        ),
                    ),
                    kwargs = {'shell': True},
                    name = sample + '_locus'
                )
            # Run jobs
            for job in loci_jobs.values():
                job.start()
            # Wait for finish
            for job in loci_jobs.values():
                job.join()
            # Make sure job worked
            for job in loci_jobs.values():
                if job.exitcode != 0:
                    raise Exception('Job {} failed with exitcode {}'.format(
                        job.name, job.exitcode
                    ))
            for sample in ['sample_1', 'sample_2']:
                print('copying results')
                _call('cp -f {}* {}'.format(prefixes[sample],
                                            prefix_long), shell=True)

        if FLAG_GENES or FLAG_GENESETS:
            print('Running gene job..')
            gene_jobs = {}
            # Create jobs
            for sample in ['sample_1', 'sample_2']:
                gene_jobs[sample] = _mp.Process(
                    target = _call,
                    args   = (
                        gene_script.format(
                            depict=depict_path, cores=cores, outdir=prefix,
                            flag_genes=FLAG_GENES, flag_genesets=FLAG_GENESETS,
                            prefix=prefixes[sample], outname=names[sample]
                        ),
                    ),
                    kwargs = {'shell': True},
                    name = sample + '_gene'
                )
            # Run jobs
            for job in gene_jobs.values():
                job.start()
            # Wait for finish
            for job in gene_jobs.values():
                job.join()
            # Make sure job worked
            for job in gene_jobs.values():
                if job.exitcode != 0:
                    raise Exception('Job {} failed with exitcode {}'.format(
                        job.name, job.exitcode
                    ))
            for sample in ['sample_1', 'sample_2']:
                _call('cp -f {}* {}'.format(prefixes[sample],
                                            prefix_long), shell=True)

        if FLAG_TISSUES:
            print('Running tissue job..')
            tissue_jobs = {}
            # Create jobs
            for sample in ['sample_1', 'sample_2']:
                tissue_jobs[sample] = _mp.Process(
                    target = _call,
                    args   = (
                        tissue_script.format(
                            depict=depict_path, cores=cores, outdir=prefix,
                            flag_genes=FLAG_GENES, flag_genesets=FLAG_GENESETS,
                            prefix=prefixes[sample], outname=names[sample]
                        ),
                    ),
                    kwargs = {'shell': True},
                    name = sample + '_tissue'
                )
            # Run jobs
            for job in tissue_jobs.values():
                job.start()
            # Wait for finish
            for job in tissue_jobs.values():
                job.join()
            # Make sure job worked
            for job in tissue_jobs.values():
                if job.exitcode != 0:
                    raise Exception('Job {} failed with exitcode {}'.format(
                        job.name, job.exitcode
                    ))
            for sample in ['sample_1', 'sample_2']:
                _call('cp -f {}* {}'.format(prefixes[sample],
                                            prefix_long), shell=True)

        # Remove temp dir as all our files are in our new dir
        #  if _os.path.abspath(prefix) != prefix_long:
            #  _call('rm -rf {}'.format(prefix), shell=True)

        # Change directory
        _os.chdir(run_path)

        # Check output files
        expected_suffices = {
            'loci': '_loci.txt',
            'gene': '_genesetenrichment.txt',
            'tissue': '_tissueenrichment.txt',
        }
        expected_outputs = {}
        for sample in ['sample_1', 'sample_2']:
            expected_outputs[sample] = {
                'loci': '{}{}'.format(prefixes[sample + '_long'],
                                    expected_suffices['loci']),
                'gene': '{}{}'.format(prefixes[sample + '_long'],
                                    expected_suffices['gene']),
                'tissue': '{}{}'.format(prefixes[sample + '_long'],
                                        expected_suffices['tissue']),
            }
        for sample, files in expected_outputs.items():
            for fl in files.values():
                assert _os.path.isfile(fl)

        with open(_pth(run_path, prefix + '_files.txt'), 'wb') as fout:
            _pickle.dump(expected_outputs, fout)

    finally:
        # Change directory
        _os.chdir(startdir)

    return expected_outputs


def run_depict_permutation(sample_1, sample_2, prefix, cores=None, perms=100,
                           run_path=None, depict_path=DEPICT,
                           perm_start=None, **fyrd_args):
    """Run DEPICT repeatedly and return locations of output files.

    This function uses fyrd to submit cluster jobs, jobs will request 2*cores
    to run, and 12G of memory.

    Takes 20 minutes to run 2 permutations on a small cluster.

    Args:
        sample_1 (str):    File name or path to file with rsids for sample 1
        sample_2 (str):    File name or path to file with rsids for sample 2
        prefix (str):      Name for the output directory, input file names will
                           be used to set output files in this directory.
        cores (int):       Number of cores to use *PER PROCESS* for DEPICT,
                           defaults to 1/2 of available cores on the machine,
                           meaning all cores will be used for run (1/2 each).
        perms (int):       Number of permutations.
        run_path (str):    Root directory to run in, defaults to current dir
        depict_path (str): Path to the DEPICT package, default set in file.
        perm_start (int):  Number to start permutations from
        fyrd_args (dict):  Fyrd keyword arguments, not required.

    Outputs:
        <prefix>/<sample_name>.geneprioritization.txt
        <prefix>/<sample_name>.loci.txt
        <prefix>/<sample_name>.tissueenrichment.txt
        <prefix>/<sample_name>.genesetenrichment.txt
        <prefix>/<sample_name>.log

    Returns:
        None: on success, else raises Exception if job fails.
    """
    if not cores:
        cores = PARAM_NCORES
    check_depict(depict_path)
    run_path = run_path if run_path else _os.path.abspath('.')
    selfpath = _os.path.realpath(__file__)
    if not perm_start:
        counts = []
        for fl in _os.listdir(run_path):
            if fl.startswith('{}_perm_'.format(prefix)):
                c = fl.split('_')[-1]
                if c.isdigit():
                    counts.append(int(c))
        if counts:
            perm_start = max(counts) + 1
        else:
            perm_start = 1
    if not isinstance(perm_start, int):
        perm_start = 1
    print('Starting permutation count at {}'.format(perm_start))
    s1_rsids = []
    s2_rsids = []
    with open(sample_1) as fin:
        s1_rsids += fin.read().strip().split('\n')
    with open(sample_2) as fin:
        s2_rsids += fin.read().strip().split('\n')
    rsids = np.array(s1_rsids + s2_rsids)
    jobs  = {}
    count = perm_start
    print('Running {} permutations'.format(perms))
    imports = ["import permute_depict as depict",
               "from permute_depict import *"]
    ttl = perms
    pbar = pb(total=ttl, unit='perms')
    while perms:
        this_perm = np.random.permutation(rsids)
        new_sample_1_data = sorted(this_perm[:len(s1_rsids)])
        new_sample_2_data = sorted(this_perm[len(s1_rsids):])
        assert len(new_sample_1_data) == len(s1_rsids)
        assert len(new_sample_2_data) == len(s2_rsids)
        perm_path = _pth(run_path, 'perm_files')
        if not _os.path.isdir(perm_path):
            _os.mkdir(perm_path)
        new_sample_1 = _pth(
            _os.path.abspath(perm_path),
            _os.path.basename(sample_1) + '_perm_{}.txt'.format(count)
        )
        new_sample_2 = _pth(
            _os.path.abspath(perm_path),
            _os.path.basename(sample_2) + '_perm_{}.txt'.format(count)
        )
        with open(new_sample_1, 'w') as fout:
            fout.write('\n'.join(new_sample_1_data))
        with open(new_sample_2, 'w') as fout:
            fout.write('\n'.join(new_sample_2_data))
        new_prefix = '{}_perm_{}'.format(prefix, count)
        job_path = _pth(run_path, 'jobs')
        if not _os.path.isdir(job_path):
            _os.mkdir(job_path)
        jobs['perm_{}'.format(count)] = (
            _fyrd.submit(
                run_depict,
                kwargs  = dict(sample_1    = new_sample_1,
                               sample_2    = new_sample_2,
                               prefix      = new_prefix,
                               cores       = cores,
                               run_path    = run_path,
                               depict_path = depict_path),
                name    = new_prefix,
                imports = ['import os as _os',
                           'from os.path import join as _pth',
                           'from subprocess import check_call as _call',
                           'import permute_depict as depict',
                           'from permute_depict import *'],
                cores   = cores*2,
                mem     = '12GB',
                scriptpath  = job_path,
                outpath     = job_path,
                runpath     = run_path,
                syspaths    = selfpath,
                **fyrd_args
            )
        )
        perms -= 1
        count += 1
        pbar.update()
    pbar.close()

    # Get output file information
    print('Permutation jobs submitted, waiting for results.')
    outputs = {}
    with pb(total=ttl, unit='results') as pbar:
        while len(outputs) < len(jobs):
            for name, job in jobs.items():
                if name in outputs:
                    continue
                job.update()
                if job.done:
                    outs = job.get()
                    outputs[name] = outs
                    with open(_pth(run_path, name) + '.files.dict', 'wb') as fout:
                        try:
                            _pickle.dump(outs, fout)
                        except TypeError:
                            pass
                    pbar.update()
            _sleep(1)

    print('Permutation jobs completed.')

    return outputs


def rescue_permutation(sample_1, sample_2, prefix, cores=None, perms=100,
                       run_path=None, depict_path=DEPICT, **fyrd_args):
    """Create output dict from run_depict_permutation from directory.

    Args:
        sample_1 (str):    File name or path to file with rsids for sample 1
        sample_2 (str):    File name or path to file with rsids for sample 2
        prefix (str):      Name for the output directory, input file names will
                           be used to set output files in this directory.
        cores (int):       Number of cores to use *PER PROCESS* for DEPICT,
                           defaults to 1/2 of available cores on the machine,
                           meaning all cores will be used for run (1/2 each).
        perms (int):       Number of permutations.
        run_path (str):    Root directory to run in, defaults to current dir
        depict_path (str): Path to the DEPICT package, default set in file.
        fyrd_args (dict):  Fyrd keyword arguments, not required.

    Outputs:
        <prefix>/<sample_name>.geneprioritization.txt
        <prefix>/<sample_name>.loci.txt
        <prefix>/<sample_name>.tissueenrichment.txt
        <prefix>/<sample_name>.genesetenrichment.txt
        <prefix>/<sample_name>.log

    Returns:
        None: on success, else raises Exception if job fails.
    """
    names = {
        'sample_1': _os.path.basename(sample_1),
        'sample_2': _os.path.basename(sample_2),
    }
    # Check output files
    expected_suffices = {
        'loci': '_loci.txt',
        'gene': '_genesetenrichment.txt',
        'tissue': '_tissueenrichment.txt',
    }

    outputs = {}

    for name in ['perm_{}'.format(i) for i in range(1, perms)]:
        count = name.split('_')[-1]
        new_prefix = '{}_perm_{}'.format(prefix, count)

        prefixes = {
            'sample_1_long': _os.path.abspath(_pth(new_prefix, names['sample_1'])),
            'sample_2_long': _os.path.abspath(_pth(new_prefix, names['sample_2'])),
            'sample_1': _pth(new_prefix, names['sample_1']),
            'sample_2': _pth(new_prefix, names['sample_2']),
        }
        outputs[name] = {}
        for sample in ['sample_1', 'sample_2']:
            outputs[name][sample] = {
                'loci': '{}{}'.format(prefixes[sample + '_long'] +
                                      '_' + name,
                                      expected_suffices['loci']),
                'gene': '{}{}'.format(prefixes[sample + '_long'] +
                                      '_' + name,
                                    expected_suffices['gene']),
                'tissue': '{}{}'.format(prefixes[sample + '_long'] +
                                        '_' + name,
                                        expected_suffices['tissue']),
            }

    return outputs


###############################################################################
#                              Manage Input data                              #
###############################################################################


def strip_hla(infile=_sys.stdin, outfile=_sys.stdout):
    """Drop any SNPs that are on chromosome 6 between 28477797 and 33448354.

    Info here:https://www.ncbi.nlm.nih.gov/grc/human/regions/MHC?asm=GRCh37

    Expects column 3 to be chromosome and column 4 to be position.
    """
    if not infile:
        infile = _sys.stdin
    if not outfile:
        outfile = _sys.stdout
    region_span = (28477797, 33448354)
    with open_zipped(infile) as fin, open_zipped(outfile, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            cols = line.strip().split('\t')
            if cols[2] == '6' or cols[2] == 'chr6':
                if region_span[0] < int(cols[3]) < region_span[1]:
                    continue
            fout.write(line)


def split_to_open_closed(infile=_sys.stdin, prefix='sample'):
    """Split an input file into two output files based on column 8.

    -1 goes to <prefix>_closed_equals_risk.txt
    1  goes to <prefix>_open_equals_risk.txt
    """
    if not infile:
        infile = _sys.stdin
    prefix = 'sample' if not prefix else prefix
    opn = []
    clsed = []
    with open_zipped(infile) as fin:
        for line in fin:
            if not line.strip():
                continue
            cols = line.strip().split('\t')
            if cols[8] == '1':
                opn.append(cols[1])
            elif cols[8] == '-1':
                clsed.append(cols[1])
    opn = sorted(set(opn), key=lambda x: int(x[2:]))
    clsed = sorted(set(clsed), key=lambda x: int(x[2:]))
    out1 = prefix + '_open_equals_risk.txt'
    out2 = prefix + '_closed_equals_risk.txt'
    with open(out1, 'w') as o1, open(out2, 'w') as o2:
        o1.write('\n'.join(opn))
        o2.write('\n'.join(clsed))


###############################################################################
#                              Helper Functions                               #
###############################################################################


def check_depict(depict_path):
    """Make sure DEPICT path is correct."""
    # Check DEPICT
    assert _os.path.isfile(
        _pth(depict_path, "LocusGenerator/LocusGenerator.jar")
    )
    assert _os.path.isfile(
        _pth(depict_path, "LocusGenerator/config.xml")
    )
    assert _os.path.isfile(
        _pth(depict_path, "Depict/Depict.jar")
    )


def open_zipped(infile, mode='r'):
    """ Return file handle of file regardless of zipped or not
        Text mode enforced for compatibility with python2 """
    mode   = mode[0] + 't'
    p2mode = mode
    if hasattr(infile, 'write'):
        return infile
    if isinstance(infile, str):
        if infile.endswith('.gz'):
            return _gzip.open(infile, mode)
        if infile.endswith('.bz2'):
            if hasattr(_bz2, 'open'):
                return _bz2.open(infile, mode)
            else:
                return _bz2.BZ2File(infile, p2mode)
        return open(infile, p2mode)


###############################################################################
#                               Run as a script                               #
###############################################################################


def command_line_parser():
    """Parse command line options.

    Returns:
        argparse parser
    """
    parser  = _argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_argparse.RawDescriptionHelpFormatter
    )

    # Subcommands
    modes = parser.add_subparsers(dest='modes')

    anlyze_depict = modes.add_parser(
        'analyze', description='Run DEPICT, plus permutation on cluster',
        help="Run DEPICT, plus permutation on cluster"
    )

    # Options
    anlyze_depict.add_argument('sample1', help="First sample")
    anlyze_depict.add_argument('sample2', help="Second sample")
    anlyze_depict.add_argument('outdir', help="Directory to write to")

    anlyze_depict.add_argument('-p', '--perms', help='Number of permutation')
    anlyze_depict.add_argument('-c', '--cores',
                                help='Cores per job, 2 jobs per queue job')
    anlyze_depict.add_argument('-r', '--run-path', help='Where to run')
    anlyze_depict.add_argument('-d', '--depict-path', help='Path to DEPICT')

    run_perms = modes.add_parser(
        'permute', description='Run DEPICT permutations on cluster',
        help="Run DEPICT permutations on cluster"
    )

    # Options
    run_perms.add_argument('sample1', help="First sample")
    run_perms.add_argument('sample2', help="Second sample")
    run_perms.add_argument('prefix',  help='Prefix to use for run')
    run_perms.add_argument('outdir',  help="Directory to write to")

    run_perms.add_argument('-p', '--perms', default=100,type=int,
                           help='Number of permutation')
    run_perms.add_argument('-c', '--cores', default=2, type=int,
                           help='Cores per job, 2 jobs per queue job')
    run_perms.add_argument('-d', '--depict-path', help='Path to DEPICT')
    run_perms.add_argument('--perm_start', type=int,
                           help='Start permutation count at this number ' +
                           'defaults to the highest count in the run dir + 1' +
                           ' or 1 if no permutations exist.')
    run_perms.add_argument('--fprofile', help='fyrd profile')
    run_perms.add_argument('--walltime', help='walltime per job')
    run_perms.add_argument('--clean', action='store_true',
                           help='clean job files')

    drop_hla = modes.add_parser(
        'drop-hla', description='Drop HLA from SNPs',
        help="Drop HLA from SNPs"
    )

    # Options
    drop_hla.add_argument('-i', '--infile', help="File to read")
    drop_hla.add_argument('-o', '--outfile', help="File to write")

    split = modes.add_parser(
        'split-input', description='Split input file based on column 8',
        help="Split input file based on column 8"
    )

    # Options
    split.add_argument('-i', '--infile', help="File to read")
    split.add_argument('-p', '--prefix', help="Prefix for outfiles")

    return parser


def main(argv=None):
    """Run as a script"""
    if not argv:
        argv = _sys.argv[1:]

    parser = command_line_parser()

    args = parser.parse_args(argv)

    if not args.modes:
        parser.print_help()
        return 0

    if args.modes == 'analyze':
        analyze_depict(args.sample1, args.sample2, args.outdir,
                       perms=args.perms, cores=args.cores,
                       run_path=args.run_path, depict_path=args.depict_path)

    if args.modes == 'permute':
        dpth = args.depict_path if args.depict_path else DEPICT
        ag = dict(
            sample_1=args.sample1, sample_2=args.sample2, prefix=args.prefix,
            cores=args.cores, perms=args.perms,
            run_path=_os.path.abspath(args.outdir),
            depict_path=_os.path.abspath(dpth)
        )
        if args.perm_start:
            ag.update(dict(perm_start=args.perm_start))
        if args.fprofile:
            ag.update(dict(profile=args.fprofile))
        if args.walltime:
            ag.update(dict(walltime=args.walltime))
        if args.clean:
            ag.update(dict(clean_outputs=True, clean_files=True))
        else:
            ag.update(dict(clean_outputs=False, clean_files=False))
        print(ag)
        run_depict_permutation(**ag)

    elif args.modes == 'drop-hla':
        strip_hla(args.infile, args.outfile)
    elif args.modes == 'split-input':
        split_to_open_closed(args.infile, args.prefix)
    else:
        print('Invalid argument {}'.format(args.modes))
        return 1


if __name__ == '__main__' and '__file__' in globals():
    _sys.exit(main())
