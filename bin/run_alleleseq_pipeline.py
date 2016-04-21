#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the core of the alleleseq pipeline.

============================================================================

          FILE: run_pipeline.py
           DIR: /scratch/users/dacre/alleleseq/newrun
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       CREATED: 2016-16-30 14:01
 Last modified: 2016-03-21 12:30

============================================================================
"""
import os
import sys
import slurmy as sl
import argparse
import pickle

MAX_JOBS = 2500
PARTITION = 'hbfraser'


def make_job_file(job, name, time, cores, mem=None, modules=[]):
    """Make a job file with 'job'."""
    modules = [modules] if isinstance(modules, str) else modules
    curdir = os.path.abspath('.')
    scrpt = os.path.join(curdir, '{}.sbatch'.format(name))
    with open(scrpt, 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        outfile.write('#SBATCH -p {}\n'.format(PARTITION))
        #  outfile.write('#SBATCH -p normal\n')
        outfile.write('#SBATCH --ntasks 1\n')
        outfile.write('#SBATCH --cpus-per-task {}\n'.format(cores))
        outfile.write('#SBATCH --time={}\n'.format(time))
        if mem:
            outfile.write('#SBATCH --mem={}\n'.format(mem))
        outfile.write('#SBATCH -o {}.o.%j\n'.format(name))
        outfile.write('#SBATCH -e {}.e.%j\n'.format(name))
        outfile.write('echo "SLURM_JOBID="$SLURM_JOBID\n')
        outfile.write('echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST\n')
        outfile.write('echo "SLURM_NNODES"=$SLURM_NNODES\n')
        outfile.write('echo "SLURMTMPDIR="$SLURMTMPDIR\n')
        outfile.write('echo "working directory = "$SLURM_SUBMIT_DIR\n')
        outfile.write('cd {}\n'.format(curdir))
        outfile.write('srun bash {}.script\n'.format(
            os.path.join(curdir, name)))
    with open(name + '.script', 'w') as outfile:
        outfile.write('#!/bin/bash\n')
        for module in modules:
            outfile.write('module load {}\n'.format(module))
        outfile.write('cd {}\n'.format(curdir))
        outfile.write('mkdir -p $LOCAL_SCRATCH\n')
        outfile.write('echo "Running {}"\n'.format(name))
        outfile.write(job + '\n')
    return scrpt


def filter_fastqs(directory):
    """Trim out all reads with 'N's.

    :directory: Directory to run in, absolute path required.
    :returns:   list of job numbers

    """
    olddir = os.path.abspath('.')
    os.chdir(directory)
    filter1 = make_job_file(
        '/home/dacre/mike_tools/bin/number_fastq_records.py ' +
        '-i in.1.fastq -o in.1.filtered.fastq',
        'filter1', '02:00:00', 1, 22000, modules='python/3.3.2')
    filter2 = make_job_file(
        '/home/dacre/mike_tools/bin/number_fastq_records.py ' +
        '-i in.2.fastq -o in.2.filtered.fastq',
        'filter2', '02:00:00', 1, 22000, modules='python/3.3.2')
    job1 = sl.monitor_submit(filter1, max_count=MAX_JOBS)
    job2 = sl.monitor_submit(filter2, max_count=MAX_JOBS)
    os.chdir(olddir)
    return [job1, job2]


def run_bowtie(directory, dependencies):
    """Run bowtie in directory.

    :dependencies: list of filter jobs for this directory.
    """
    olddir = os.path.abspath('.')
    os.chdir(directory)
    bowtie1 = make_job_file('bowtie --best --strata ' +
                            '-p 16 --chunkmbs 2000 --maxins 2000 -m 1 -q ' +
                            '../pat/PatRef ' +
                            '--chunkmbs 2000 --maxins 1000' +
                            '-1 in.1.filtered.fastq -2 in.2.filtered.fastq ' +
                            'pat_alignment.bam 2> pat_alignment.log',
                            'pat_bowtie', '24:00:00', 16, modules=['bowtie1'])
    bowtie2 = make_job_file('bowtie --best --strata ' +
                            '-p 16 --chunkmbs 2000 --maxins 2000 -m 1 -q ' +
                            '../mat/MatRef ' +
                            '--chunkmbs 2000 --maxins 1000' +
                            '-1 in.1.filtered.fastq -2 in.2.filtered.fastq ' +
                            'mat_alignment.bam 2> mat_alignment.log',
                            'mat_bowtie', '24:00:00', 16, modules=['bowtie1'])
    job1 = sl.monitor_submit(bowtie1, dependencies, max_count=MAX_JOBS)
    job2 = sl.monitor_submit(bowtie2, dependencies, max_count=MAX_JOBS)
    os.chdir(olddir)
    return [job1, job2]


def run_star(directory, dependencies):
    """Run STAR in directory.

    :dependencies: list of filter jobs for this directory.
    """
    olddir = os.path.abspath('.')
    os.chdir(directory)
    unzip = ('cat in.1.fastq.gz | /home/dacre/usr/bin/unpigz -p 16 > in.1.fastq\n' +
             'cat in.2.fastq.gz | /home/dacre/usr/bin/unpigz -p 16 > in.2.fastq\n')
    star1 = ('/home/dacre/usr/bin/STAR --runThreadN 16 ' +
             '--genomeDir ../pat/pat_star ' +
             #  '--readFilesIn in.1.fastq in.2.fastq ' +
             '--readFilesIn in.1.filtered.fastq in.2.filtered.fastq ' +
             '--outFilterMultimapNmax 1 ' +
             '--outFileNamePrefix pat_alignment_ ' +
             '--outSAMtype BAM SortedByCoordinate ' +
             '--outSAMattributes MD NH ' +
             '--clip5pNbases 6')
    star2 = ('/home/dacre/usr/bin/STAR --runThreadN 16 ' +
             '--genomeDir ../mat/mat_star ' +
             #  '--readFilesIn in.1.fastq in.2.fastq ' +
             '--readFilesIn in.1.filtered.fastq in.2.filtered.fastq ' +
             '--outFilterMultimapNmax 1 ' +
             '--outFileNamePrefix mat_alignment_ ' +
             '--outSAMtype BAM SortedByCoordinate ' +
             '--outSAMattributes MD NH ' +
             '--clip5pNbases 6')
    #  if not os.path.exists(os.path.join(directory, 'in.1.fastq')):
        #  unzip = make_job_file(unzip, 'unzip', '04:00:00', 16)
        #  unzip_job = sl.monitor_submit(unzip, dependencies, max_count=MAX_JOBS)
        #  dependencies = unzip_job
    star1 = make_job_file(star1, 'pat_star', '12:00:00', 16, modules=['STAR'])
    star2 = make_job_file(star2, 'mat_star', '12:00:00', 16, modules=['STAR'])
    job1 = sl.monitor_submit(star1, dependencies, max_count=MAX_JOBS)
    job2 = sl.monitor_submit(star2, dependencies, max_count=MAX_JOBS)
    os.chdir(olddir)
    return [job1, job2]


def clean_bowtie(directory, dependencies):
    olddir = os.path.abspath('.')
    os.chdir(directory)
    bowtie1 = make_job_file('samtools sort pat_alignment.bam pat_sorted\n' +
                            'samtools view pat_sorted.bam > ' +
                            'pat_alignment.sam\n' +
                            'rm pat_alignment.bam pat_sorted.bam',
                            'pat_clean', '08:00:00', 2, modules=['samtools'])
    bowtie2 = make_job_file('samtools sort mat_alignment.bam mat_sorted\n' +
                            'samtools view mat_sorted.bam > ' +
                            'mat_alignment.sam\n' +
                            'rm mat_alignment.bam mat_sorted.bam',
                            'mat_clean', '08:00:00', 1, modules=['samtools'])
    job1 = sl.monitor_submit(bowtie1, dependencies, max_count=MAX_JOBS)
    job2 = sl.monitor_submit(bowtie2, dependencies, max_count=MAX_JOBS)
    os.chdir(olddir)
    return [job1, job2]


def clean_star(directory, dependencies):
    olddir = os.path.abspath('.')
    os.chdir(directory)
    clean1 = make_job_file('samtools view ' +
                           'pat_alignment_Aligned.sortedByCoord.out.bam' +
                           ' > pat_alignment.sam',
                           'pat_clean', '08:00:00', 2, mem=10000,
                           modules=['samtools'])
    clean2 = make_job_file('samtools view ' +
                           'mat_alignment_Aligned.sortedByCoord.out.bam' +
                           ' > mat_alignment.sam',
                           'mat_clean', '08:00:00', 2, mem=10000,
                           modules=['samtools'])
    job1 = sl.monitor_submit(clean1, dependencies, max_count=MAX_JOBS)
    job2 = sl.monitor_submit(clean2, dependencies, max_count=MAX_JOBS)
    os.chdir(olddir)
    return [job1, job2]


def merge(directory, dependencies, type):
    """Run AlleleSeq Merge Step."""
    olddir = os.path.abspath('.')
    os.chdir(directory)
    merge1 = make_job_file('python ' +
                           '../../AlleleSeq_pipeline_v1.2a/MergeBowtie.py ' +
                           'pat_alignment.sam mat_alignment.sam ' +
                           '../genome/%s_' + type + '.map > ' +
                           'merged_reads.sam 2> merged_reads.log', 'merge',
                           '06:00:00', 1, 8000, modules='python/2.7.5')
    job1 = sl.monitor_submit(merge1, dependencies, max_count=MAX_JOBS)
    os.chdir(olddir)
    return job1


def count(directory, dependencies, type):
    """Run AlleleSeq Count Step."""
    olddir = os.path.abspath('.')
    os.chdir(directory)
    name = os.path.basename(directory)
    count1 = make_job_file('python ' +
                           '../../AlleleSeq_pipeline_v1.2a/SnpCounts.py ' +
                           '../*snps.txt merged_reads.sam ' +
                           '../genome/%s_' + type +
                           '.map {}.cnt '.format(name),
                           'count', '06:00:00', 4, 16000,
                           modules='python/2.7.5')
    job1 = sl.monitor_submit(count1, dependencies, max_count=MAX_JOBS)
    os.chdir(olddir)
    return job1


def run_pipeline(step):
    """Run the complete pipeline here."""
    bxc_list = []
    cxb_list = []
    for i in os.listdir('BxC'):
        if os.path.isdir(os.path.join('BxC', i)):
            if i.isdigit or i[2:].isdigit:
                if i == 'mat' or i == 'pat' or i == 'genome':
                    continue
                bxc_list.append(os.path.join('BxC', i))
    for i in os.listdir('CxB'):
        if os.path.isdir(os.path.join('CxB', i)):
            if i.isdigit or i[2:].isdigit:
                if i == 'mat' or i == 'pat' or i == 'genome':
                    continue
                cxb_list.append(os.path.join('CxB', i))
    bxc = {}
    cxb = {}
    # Make fastas
    if step == 1:
        print("Submitting FastQ Filter")
        for readdir in bxc_list:
            bxc[readdir] = {'filter':
                            filter_fastqs(readdir)}
        for readdir in cxb_list:
            cxb[readdir] = {'filter':
                            filter_fastqs(readdir)}
        step += 1
    # Run bowtie
    if step == 2:
        print("Submitting mapping jobs")
        for readdir in bxc_list:
            #  bxc[readdir]['bowtie'] = run_bowtie(readdir,
                                                #  bxc[readdir]['filter'])
            try:
                bxc[readdir]['star'] = run_star(readdir,
                                                bxc[readdir]['filter'])
            except KeyError:
                bxc[readdir] = {}
                bxc[readdir]['star'] = run_star(readdir,
                                                None)
        for readdir in cxb_list:
            #  cxb[readdir]['bowtie'] = run_bowtie(readdir,
                                                #  cxb[readdir]['filter'])
            try:
                cxb[readdir]['star'] = run_star(readdir,
                                                cxb[readdir]['filter'])
            except KeyError:
                cxb[readdir] = {}
                cxb[readdir]['star'] = run_star(readdir,
                                                None)
        step += 1
    if step == 3:
        print("Submitting cleaning jobs")
        for readdir in bxc_list:
            #  bxc[readdir]['samtools'] = clean_bowtie(readdir,
                                            #  bxc[readdir]['bowtie'])
            try:
                bxc[readdir]['clean'] = clean_star(readdir,
                                                   bxc[readdir]['star'])
            except KeyError:
                bxc[readdir] = {}
                bxc[readdir]['clean'] = clean_star(readdir, None)
        for readdir in cxb_list:
            #  cxb[readdir]['samtools'] = clean_bowtie(readdir,
                                            #  cxb[readdir]['bowtie'])
            try:
                cxb[readdir]['clean'] = clean_star(readdir,
                                                   cxb[readdir]['star'])
            except KeyError:
                cxb[readdir] = {}
                cxb[readdir]['clean'] = clean_star(readdir, None)
        step += 1
    # Merge bowtie
    if step == 4:
        print("Submitting AlleleSeq bowtie mat/pat merge")
        for readdir in bxc_list:
            try:
                bxc[readdir]['merge'] = merge(readdir,
                                              bxc[readdir]['clean'], 'BxC')
            except KeyError:
                bxc[readdir] = {}
                bxc[readdir]['merge'] = merge(readdir, None, 'BxC')
        for readdir in cxb_list:
            try:
                cxb[readdir]['merge'] = merge(readdir,
                                              cxb[readdir]['clean'], 'CxB')
            except KeyError:
                cxb[readdir] = {}
                cxb[readdir]['merge'] = merge(readdir, None, 'CxB')
        step += 1
    # Count reads
    if step == 5:
        print("Submitting AlleleSeq allele counting script")
        for readdir in bxc_list:
            try:
                bxc[readdir]['count'] = count(readdir,
                                              bxc[readdir]['merge'], 'BxC')
            except KeyError:
                bxc[readdir] = {}
                bxc[readdir]['count'] = count(readdir, None, 'BxC')
        for readdir in cxb_list:
            try:
                cxb[readdir]['count'] = count(readdir,
                                              cxb[readdir]['merge'], 'CxB')
            except KeyError:
                cxb[readdir] = {}
                cxb[readdir]['count'] = count(readdir, None, 'CxB')

    print("Job mapping:")
    with open('alleleseq_jobs.pickle', 'wb') as outf:
        pickle.dump([bxc, cxb], outf)
    print(bxc)
    print(cxb)


def main(argv=None):
    """Run as a script."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument('--step', choices=[1, 2, 3, 4, 5], default=1, type=int,
                        help='Step to start from. 1=filter, 2=mapping, ' +
                        '3=Map clean, 4=Merging, 5=Counting.')

    args = parser.parse_args(argv)

    run_pipeline(args.step)

    return 0

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
