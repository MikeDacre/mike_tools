#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the entire WASP pipeline with either tophat or STAR.

============================================================================

          FILE: wasp_pipeline.py
           DIR: /scratch/users/dacre
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.2
       CREATED: 2016-07-12 09:02
 Last modified: 2016-03-07 11:27

============================================================================
"""
import os
import sys
import argparse
import slurmy
import pickle
import logme

MAX_JOBS  = 2000
PARTITION = 'owners'
logme.MIN_LEVEL = 'info'  # Change to debug for verbose logging
logme.LOGFILE   = 'wasp_submit_log.log'


def make_job_file(job, name, time, cores, mem=None, modules=[]):
    """Make a job file with 'job'."""
    modules = [modules] if isinstance(modules, str) else modules
    curdir = os.path.abspath('.')
    print(name)
    scrpt = os.path.join(curdir, '{}.sbatch'.format(name))
    with open(scrpt, 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('#SBATCH -p {}\n'.format(PARTITION))
        outfile.write('#SBATCH --ntasks 1\n')
        outfile.write('#SBATCH --cpus-per-task {}\n'.format(cores))
        outfile.write('#SBATCH --time={}\n'.format(time))
        if mem:
            outfile.write('#SBATCH --mem={}\n'.format(mem))
        outfile.write('#SBATCH -o {}.o.%j\n'.format(name))
        outfile.write('#SBATCH -e {}.e.%j\n'.format(name))
        outfile.write('cd {}\n'.format(curdir))
        outfile.write('srun bash {}.script\n'.format(
            os.path.join(curdir, name)))
    with open(name + '.script', 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('echo "SLURM_JOBID="$SLURM_JOBID\n')
        outfile.write('echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST\n')
        outfile.write('echo "SLURM_NNODES"=$SLURM_NNODES\n')
        outfile.write('echo "SLURMTMPDIR="$SLURMTMPDIR\n')
        outfile.write('echo "working directory = "$SLURM_SUBMIT_DIR\n')
        for module in modules:
            outfile.write('module load {}\n'.format(module))
        outfile.write('cd {}\n'.format(curdir))
        outfile.write('mkdir -p $LOCAL_SCRATCH\n')
        outfile.write('echo "Running {}"\n'.format(name))
        outfile.write(job + '\n')
    return scrpt


def run_mapping(name, infiles, genome, algorithm='STAR', gtf=None,
                dependency=None):
    """Run read mapping using either tophat or STAR.

    :name:       A name prefix to use for the output.
    :infiles:    List of fastqs, space separated for paired end, comma
                 separated for batches. Must be a string.
                 Note: if gzipped and using STAR, they will be unzipped
                 and rezipped during mapping
    :genome:     The genome or STAR genome index.
    :algorithm:  STAR or tophat. Case ignored.
    :gtf:        A GTF of genes for tophat, not required.
    :dependency: The job number of the remapping step.
    :returns:    Job number of mapping step and name of output bam.

    """
    if algorithm.lower() == 'star':
        cmnd     = []
        new_list = []
        zipped   = False
        for fl in infiles.split(' '):
            b = []
            for i in fl.split(','):
                if i.endswith('.gz'):
                    zipped = True
                    cmnd.append('/home/dacre/usr/bin/unpigz -p 16 ' + i)
                    b.append(i[:-3])
                else:
                    b.append(i)
            new_list.append(','.join(b))
        infiles = ' '.join(new_list)

        cmnd.append('/home/dacre/usr/bin/STAR --runThreadN 16 ' +
                    '--genomeDir {} '.format(genome) +
                    '--readFilesIn {} '.format(infiles) +
                    '--outFilterMultimapNmax 1 ' +
                    '--outFileNamePrefix {} '.format(name) +
                    '--outSAMtype BAM SortedByCoordinate ' +
                    '--outSAMattributes MD NH ' +
                    '--clip5pNbases 6')

        if zipped:
            for fl in new_list:
                for i in fl.split(','):
                    cmnd.append(
                        '/home/dacre/usr/bin/pigz -p 16 {}'.format(i))

        command = '\n'.join(cmnd)
        outbam  = name + 'Aligned.sortedByCoord.out.bam'
        modules = ['STAR']

    elif algorithm.lower() == 'tophat':
        command = 'tophat --microexon-search -o {}'.format(name + '_tophat')
        command = command + ' -G ' + gtf if gtf else command
        command = command + ' -p 16 {} {}\n'.format(genome, infiles)
        outbam  = name + '_accepted_hits.bam'
        command = command + 'mv {}/accepted_hits.bam {}'.format(
            name + '_tophat', outbam)
        modules = ['python/2.7.5', 'tophat']

    else:
        raise Exception('Invalid algorithm: {}'.format(algorithm))

    return (slurmy.monitor_submit(make_job_file(command, name,
                                                '24:00:00', 16,
                                                modules=modules),
                                  dependency, MAX_JOBS),
            outbam)


def wasp_step_1(fl, snp_dir, pipeline=None, dependency=None):
    """Run find_intersecting_snps.py on fl.

    :fl:         The sam or bam file to run on.
    :snp_dir:    The SNP directory required by WASP.
    :pipeline:   The path to the WASP pipeline.
    :dependency: The job number of the remapping step.
    :returns:    The job number.
    """
    command = os.path.join(os.path.abspath(pipeline),
                           'find_intersecting_snps.py') \
        if pipeline else 'find_intersecting_snps.py'
    logme.log('Submitting wasp step 1 for {}'.format(fl), level='debug')
    return slurmy.monitor_submit(make_job_file(
        'python2 {} -m 1000000 {} {}'.format(command, fl, snp_dir),
        fl + '_step1', '06:00:00', 4, '20000', modules=['python/2.7.5']),
        dependency, MAX_JOBS)


def wasp_step_2(name, remapped, pipeline=None, dependency=None):
    """Run filter_remapped_reads.py following second mapping.

    :name:       The name of the original mapped bam or sam, used to make file
                 names
    :remapped:   The file created by the second mapping.
    :pipeline:   The path to the WASP pipeline.
    :dependency: The job number of the remapping step.
    :returns:    The job number.

    """
    command = os.path.join(os.path.abspath(pipeline),
                           'filter_remapped_reads.py') \
        if pipeline else 'filter_remapped_reads.py'
    # Trim the name
    shortname = '.'.join(name.split('.')[:-1]) if name.endswith('.bam') \
        or name.endswith('.sam') else name
    logme.log('Submitting wasp step 2 for {}'.format(shortname), level='debug')
    return slurmy.monitor_submit(make_job_file(
        'python2 {} {} {} {} {}'.format(command,
                                        shortname + '.to.remap.bam',
                                        remapped,
                                        shortname + '.remap.keep.bam',
                                        shortname + '.to.remap.num.gz'),
        shortname + '_step2', '16:00:00', 4, '20000',
        modules=['python/2.7.5']), dependency, MAX_JOBS)


def merge_bams(name, dependency=None):
    """Use samtools to merge two bam files."""
    shortname = '.'.join(name.split('.')[:-1]) if name.endswith('.bam') \
        or name.endswith('.sam') else name
    orig_reads = shortname + '.keep.bam'
    remapped   = shortname + '.remap.keep.bam'
    uname      = shortname + '_wasp_final_unsorted.bam'
    final_name = shortname + '_wasp_final.bam'
    return slurmy.monitor_submit(make_job_file(
        'samtools merge -f {} {} {}\n'.format(uname, orig_reads, remapped) +
        'samtools sort -o {} {}'.format(final_name, uname),
        shortname + '_merge', '12:00:00', 1, '16000', 'samtools'),
        dependency, MAX_JOBS)


def run_wasp(files, snp_dir, genome, algorithm='star', gtf=None, pipeline=None,
             step=1, remapped_bam=None):
    """Run the complete WASP pipeline.

    :files:     All the files to run on, can be fastq or sam/bam. If fastq, or
                directory, an initial mapping is done.
    :snp_dir:   The SNP directory required by WASP.
    :genome:    A genome directory for tophat, or the STAR index directory for
                STAR
    :algorithm: 'star' or 'tophat'
    :gtf:       A GTF of genes for tophat, not required.
    :pipeline:  The location of the WASP pipeline
    :step:      Start at steps 1, 2, 3, or 4 instead of at the beginning,
                ignored if files are fastq.
    :returns:   None.

    """
    all_jobs  = {}
    save_file = 'wasp_jobs.pickle'
    # Detect if need to run mapping
    if files[0].endswith('.fq') or files[0].endswith('.fastq') \
            or os.path.isdir(files[0]):
        logme.log('File contains fastq, running initial mapping',
                  also_write='stderr')
        initial_map = True
    else:
        initial_map = False

    initial_step = step
    if step == 2:
        step_1 = None
    elif step == 3:
        remapped = None
        remap    = None
    elif step == 4:
        step_2 = None

    # Loop through every file and run all steps of the pipeline.
    for fl in files:
        step = initial_step
        map_job = None

        # Initial mapping
        if initial_map:
            if os.path.isdir(fl):
                fl = []
                for i in os.listdir(fl):
                    if os.path.isfile(i):
                        if 'fq' in i.split('.') \
                                or 'fastq' in i.split('.'):
                            fl.append(os.path.join(fl, i))
                single = []
                pair_1 = []
                pair_2 = []
                for i in fl:
                    if '_1' in i:
                        pair_1.append(i)
                    elif '_2' in i:
                        pair_2.append(i)
                    else:
                        single.append(i)
                if single and pair_1 or pair_2:
                    raise Exception('Cannot have both single and paired')
                if single:
                    map_files = ','.join(single)
                else:
                    map_files = ' '.join([','.join(pair_1),
                                          ','.join(pair_2)])
            else:
                map_files = fl
            map_job, bamfile = run_mapping(fl + '_map1', map_files, genome,
                                           algorithm, gtf)
            fl = bamfile  # Switch back to normal mode
            all_jobs[map_job] = fl + '_map1'
            with open(save_file, 'wb') as outf:
                pickle.dump(all_jobs, outf)

        # WASP 1
        if step == 1:
            step_1 = wasp_step_1(fl, snp_dir, pipeline, map_job)
            logme.log('{} WASP step 1: {}'.format(fl, step_1))
            step += 1
            all_jobs[step_1] = fl + '_step1'
            with open(save_file, 'wb') as outf:
                pickle.dump(all_jobs, outf)

        # Remapping
        if step == 2:
            readfile = '.'.join(fl.split('.')[:-1]) + '.remap.fq.gz'
            remap, remapped = run_mapping(fl.split('_')[0] + '_remap',
                                          readfile, genome, algorithm, gtf,
                                          step_1)
            logme.log('{} Remapping: {}'.format(fl, remap))
            step += 1
            all_jobs[remap] = fl + '_remap'
            with open(save_file, 'wb') as outf:
                pickle.dump(all_jobs, outf)

        # WASP 2
        if step == 3:
            if not remapped:
                remapped = fl + '_remapAligned.sortedByCoord.out.bam'
            step_2 = wasp_step_2(fl, remapped, pipeline, remap)
            logme.log('{} WASP step 2: {}'.format(fl, step_2))
            step += 1
            all_jobs[step_2] = fl + '_step2'
            with open(save_file, 'wb') as outf:
                pickle.dump(all_jobs, outf)

        # Merge Files
        if step == 4:
            merge_job = merge_bams(fl, step_2)
            logme.log('{} Merge Step: {}'.format(fl, merge_job))
            all_jobs[merge_job] = fl + '_merge'
            with open(save_file, 'wb') as outf:
                pickle.dump(all_jobs, outf)

    return 0


def main(argv=None):
    """Run as a script."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('files', nargs='+',
                        help="Input files, either .fq or .sam/.bam")

    parser.add_argument('-w', '--wasp', default=None,
                        help="WASP pipeline location.")
    parser.add_argument('-s', '--snp_dir', required=True,
                        help="The SNP directory required for WASP")

    # Alignment algorithm choice
    alg = parser.add_argument_group('Alignment Algorithm Choice')
    alg.add_argument('-a', '--algorithm', choices=['star', 'tophat'],
                     default='star', help="star or tophat, default star")
    alg.add_argument('-g', '--genome', required=True,
                     help="Genome to align against, STAR index for STAR")
    alg.add_argument('--gtf', default=None,
                     help="GTF Gene file, only used by tophat, not required.")

    # For running step 4 again
    # Arguments
    wasp_2 = parser.add_argument_group('Resuming pipeline')
    wasp_2.add_argument('--step', choices=[1, 2, 3, 4], default=1, type=int,
                        help="Start pipeline from this step. Ignored if " +
                        "files are fastqs.")
    wasp_2.add_argument('--remapped_bam',
                        help=("The bam file suffix from the remapping step, " +
                              "used if starting from step 3."))

    # Logging
    logging = parser.add_argument_group('Logging')
    logging.add_argument('-l', '--logfile', default=sys.stderr,
                         help="Log File, Default STDERR")
    logging.add_argument('-v', '--verbose', action='store_true',
                         help="Verbose logging.")

    args = parser.parse_args(argv)

    # Set up logging
    if args.verbose:
        logme.MIN_LEVEL = 'debug'
    logme.LOGFILE = args.logfile

    # Run the pipeline
    gtf_file = os.path.abspath(args.gtf) if args.gtf else None
    return run_wasp(args.files, os.path.abspath(args.snp_dir),
                    os.path.abspath(args.genome), args.algorithm,
                    gtf_file, os.path.abspath(args.wasp),
                    args.step, args.remapped_bam)


if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
