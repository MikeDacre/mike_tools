WASP Pipeline
=============

A wrapper to run [WASP](https://github.com/bmvdgeijn/WASP) with either
[STAR](https://github.com/alexdobin/STAR) or
[tophat](https://ccb.jhu.edu/software/tophat/index.shtml). Can start with
either raw fastq read (gzipped is fine) or with initial bam files.

Options
-------

    usage: wasp_pipeline.py [-h] [-w WASP] -s SNP_DIR [-a {star,tophat}] -g GENOME
                            [--gtf GTF] [--step {1,2,3,4}]
                            [--remapped_bam REMAPPED_BAM] [-l LOGFILE] [-v]
                            files [files ...]

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

    positional arguments:
      files                 Input files, either .fq or .sam/.bam

    optional arguments:
      -h, --help            show this help message and exit
      -w WASP, --wasp WASP  WASP pipeline location.
      -s SNP_DIR, --snp_dir SNP_DIR
                            The SNP directory required for WASP

    Alignment Algorithm Choice:
      -a {star,tophat}, --algorithm {star,tophat}
                            star or tophat, default star
      -g GENOME, --genome GENOME
                            Genome to align against, STAR index for STAR
      --gtf GTF             GTF Gene file, only used by tophat, not required.

    Resuming pipeline:
      --step {1,2,3,4}      Start pipeline from this step. Ignored if files are
                            fastqs.
      --remapped_bam REMAPPED_BAM
                            The bam file suffix from the remapping step, used if
                            starting from step 3.

    Logging:
      -l LOGFILE, --logfile LOGFILE
                            Log File, Default STDERR
      -v, --verbose         Verbose logging.

