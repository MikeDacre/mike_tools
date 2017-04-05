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
import sys as _sys
from os.path import join as _pth

from subprocess import check_call as _call
import multiprocessing as _mp

import numpy as np
import scipy as sp
import scipy.stats as sts
import pandas as pd

import fyrd as _fyrd

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
PARAM_NCORES  = 2  # Number of cores to use *PER PROCESS* for DEPICT

###############################################################################
#                          Do Not Modify Below Here                           #
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
        None: on success, else raises Exception if job fails.
    """
    # Get DEPICT dir
    if not _os.path.isdir(depict_path):
        depict_path = DEPICT
    depict_path = _os.path.abspath(depict_path)

    check_depict(depict_path)

    # Change directory
    startdir = _os.path.curdir
    _os.chdir(run_path)

    # Check sample files
    infiles = {
        'sample_1': _os.path.abspath(sample_1),
        'sample_2': _os.path.abspath(sample_2),
    }
    for sample in infiles.values():
        assert _os.path.isfile(sample)
        with open(sample) as fin:
            assert fin.readline().strip().startswith('rs')

    # Get names
    names = {
        'sample_1': '.'.join(_os.path.basename(sample_1).split('.')[:-1]),
        'sample_2': '.'.join(_os.path.basename(sample_2).split('.')[:-1]),
    }
    prefixes = {
        'sample_1_long': _os.path.abspath(_pth(prefix, names['sample_1'])),
        'sample_2_long': _os.path.abspath(_pth(prefix, names['sample_2'])),
        'sample_1': _pth(prefix, names['sample_1']),
        'sample_2': _pth(prefix, names['sample_2']),
    }

    # Set cores
    if not cores:
        cores = int(_mp.cpu_count()/2)

    # Create directory
    prefix_long = _os.path.abspath(prefix)
    if not _os.path.isdir(prefix_long):
        _os.makedirs(prefix_long)

    # Change directory
    _os.chdir(depict_path)
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
    if _os.path.abspath(prefix) != prefix_long:
        _call('rm -rf {}'.format(prefix), shell=True)

    # Change directory
    _os.chdir(startdir)


def permute_depict(sample_1, sample_2, prefix, cores=None, perms=100,
                   run_path=None, depict_path=DEPICT, **fyrd_args):
    """Run DEPICT twice, once on each sample, DEPICT will be run in parallel.

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
        None: on success, else raises Exception if job fails.
    """
    check_depict(depict_path)
    rsids = []
    for sample in [sample_1, sample_2]:
        with open(sample) as fin:
            rsids += fin.read().strip().split('\n')
    rsids = np.array(rsids)
    jobs  = []
    count = 1
    while perms:
        this_perm = np.random.permutation(rsids)
        half_len = int(len(this_perm)/2)
        new_sample_1_data = sorted(this_perm[:half_len])
        new_sample_2_data = sorted(this_perm[half_len:])
        assert len(new_sample_1_data) == len(new_sample_2_data)
        new_sample_1 = sample_1 + '_perm_{}'.format(count)
        new_sample_2 = sample_2 + '_perm_{}'.format(count)
        with open(new_sample_1, 'w') as fout:
            fout.write('\n'.join(new_sample_1_data))
        with open(new_sample_2, 'w') as fout:
            fout.write('\n'.join(new_sample_2_data))
        new_prefix = '{}_perm_{}'.format(prefix, count)
        jobs.append(
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
                           'from subprocess import check_call as _call'],
                cores   = 8,
                mem     = '12GB',
                **fyrd_args
            )
        )
        perms -= 1
        count += 1
    for job in jobs:
        job.wait()


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
