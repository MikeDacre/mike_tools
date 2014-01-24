#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Copyright © Mike Dacre <mike.dacre@gmail.com>
#
# Distributed under terms of the MIT license
"""
================================================================================================

          FILE: vcf_simplify (python 3) (multithreading)
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, Property of Stanford, Use however you wish
       VERSION: 0.3
       CREATED: 2014-01-21 17:38
 Last modified: 2014-01-23 16:36

   DESCRIPTION: Take a compressed vcf file such as
                ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/ALL.chr1.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz
                and create a simplified matrix where all genotypes are represented
                as 0/1/2 where 0: homozygote_1; 1:heterozygote; 2: homozygote_2.

                Out file format (tab delimited):
                CHR(e.g.1/MT)\\tPOS(e.g.10583)\\tSNP_ID(e.g.rs58108140)\\tref\\talt\\tqual\\tfilter\\t[sample_1]\\t[sample_2]\\t...\\t[sample_n]

                Execution time on a single 1000genomes file is
                2647.92s user 9.00s system 97% cpu 45:16.76 total

                Additionally, it is possible to filter a 1000genomes style vcf file, or 
                a previously simplified vcf file by population, region, or platform.
                Filtering requires a 1000genomes style panel file, such as:
                ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/integrated_call_samples.20101123.ALL.panel

          NOTE: The vcf files must be gzipped, and the genotypes must be encoded as
                0|0, 0|1, 1|0, 1|1

USAGE EXAMPLES: Simply vcf files:
                ./vcf_simplify.py ALL.chr1.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz\\
                ALL.chr2.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz

                Filter 1000genomes vcf files by population:
                ./vcf_simplify.py -p integrated_call_samples.20101123.ALL.panel\\
                --filter_population CEU,YRI \\
                ALL.chr1.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz\\
                ALL.chr2.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz

=============================================================================================
"""
import gzip, sys, re
from os import path
from multiprocessing import Pool

# Default threads
default_threads = 8

def vcf_simplify(vcf_file, output_directory='.', logfile=sys.stderr, verbose=False):
    """Take a 1000genomes style vcf file (MUST BE GZIPPED) and
       simplify it to:

       chr\\tpos\\trsID\\tref\\talt\\tqual\\tfilter\\tsample_1\\t[sample_2]\\t...[sample_n]\\n"""


    # Get an outfile name:
    outfile = re.sub('.vcf.gz$','', path.basename(vcf_file)) + '_simplified.vcf.gz'
    outfile = path.realpath(output_directory) + '/' + outfile

    with gzip.open(vcf_file, 'rb') as infile:

        # Check file format
        if not infile.readline().decode('utf8').rstrip() == '##fileformat=VCFv4.1':
            _logme(' '.join(["\n\nERROR: File", vcf_file, "does not have '##fileformat=VCFv4.1'",
                            "as its first line, not processing.\n\n"]), logfile, 2)
            return

        # Ignore comment lines and get header
        header = ''
        while 1:
            h = infile.readline().decode('utf8').rstrip()
            if h.startswith('#CHROM'):
                header = h
                break
            else:
                continue

        # Parse header
        h = header.split('\t')
        if not h[0:7] == ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER']:
            _logme(' '.join(["ERROR:",  vcf_file, "header is\n", h, "\nit should be\n",
                                "['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER']\n",
                                "Aborting\n\n"]), logfile, 2)
            return

        # Parse the rest of the file and print the output
        with gzip.open(outfile, 'wb') as output:
            # Print header line
            output.write(''.join(['\t'.join(h[0:7]), '\t', '\t'.join(h[9:]), '\n']).encode('utf8'))

            while infile:
                line = infile.readline().decode('utf8').rstrip()

                if line:
                    fields = line.split('\t')
                else:
                    break

                # Get initial columns
                outstring = fields[0:7]

                # Parse individuals
                for individual in range(9, len(fields)):
                    genotype = fields[individual].split(':')[0]
                    if genotype == '0|0':
                        # Homozygote 1
                        outstring.append('0')
                    elif genotype == '0|1' or genotype == '1|0':
                        # Heterozygote
                        outstring.append('1')
                    elif genotype == '1|1':
                        # Homozygote 2
                        outstring.append('2')
                    else:
                        # Throw ERROR
                        error_string = ''.join(["ERROR: Individual ", str(individual),
                                        " in SNP ", fields[2], "did not have a known genotype.\n",
                                        "Reported genotype was: ", genotype])
                        _logme(error_string, logfile, 2)
                        raise Exception(error_string)

                # Print line
                output.write(''.join(['\t'.join(outstring), '\n']).encode('utf8'))

def parse_panel_file(panel_file, logfile=sys.stderr, verbose=False):
    """Take 1000genomes-style panel file and return a dictionary
       with SampleID->(population, region, [platforms])
       1000genomes panel files have no header, and have the following
       columns:

       SampleID\tPopulation\tRegion\tPlatform\n

       e.g.:
       ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/integrated_call_samples.20101123.ALL.panel"""

    # Open logfile
    if isinstance(logfile, str):
        logfile = open(logfile, 'a')

    sample_info = {}
    with open(panel_file, 'r') as infile:
        for line in infile:
            fields = line.rstrip().split('\t')
            if fields[0] in sample_info:
                print_level = 2 if verbose else 0
                _logme("Fields[0] is duplicated", logfile, print_level)
            else:
                sample_info[fields[0]] = (fields[1], fields[2], [re.sub(',', '', platforms) for platforms in fields[3:]])

    return(sample_info)

def filter(file_list, population_list, panel_file, output_directory='.', threads=default_threads, verbose=False, logfile=sys.stderr):
    """Filter provided vcf files based on population.
       Works on either original or simplified vcf files
       Works threaded, uses the _filter private function for
       actual processing"""

    # Get full path to output directory
    output_directory = path.realpath(output_directory)

    # Get sample info from panel file
    sample_info = parse_panel_file(panel_file, logfile=logfile, verbose=verbose)

    # Sort out some threads
    pool = Pool(processes=int(threads))
    running_processes = []

    # Queue up the private function threads
    for vcf_file in file_list:
        running_processes.append(pool.apply_async(_filter, (vcf_file, population_list, sample_info, output_directory)))

    # Run the threads
    for process in running_processes:
        process.get()

def _filter(vcf_file, population_list, sample_info, output_directory):
    """A private function to run the meat of the filtering
       Requires gzipped files like everything else"""
     
    # Get an outfile name:
    outfile = re.sub('.vcf.gz$','', path.basename(vcf_file)) + '_' + '_'.join(population_list) + '.vcf.gz'
    outfile = path.realpath(output_directory) + '/' + outfile
    
    with gzip.open(vcf_file, 'rb') as infile:
        # Check if this is 1000genomes or simplified
        header      = ''
        simplified  = False
        line1 = infile.readline().decode('utf8').rstrip()
        if line1 == '##fileformat=VCFv4.1':
            # Is 1000genomes file
            while 1:
                h = infile.readline().decode('utf8').rstrip()
                if h.startswith('#CHROM'):
                    header      = h
                    simplified  = False
                    break
                elif h.startswith('##'):
                    continue  
                else:
                    error_string = "File: " + vcf_file + " Appears to be an invalid 1000genomes file"
                    _logme(error_string, sys.stderr, 2)
                    raise Exception(error_string)

        elif line1.startswith('#CHROM'):
            # Is simplified file
            header      = line1
            simplified  = True
        else:
            error_string = "File: " + vcf_file + " is neither a 1000genomes file nor a simplified file"
            _logme(error_string, sys.stderr, 2)
            raise Exception(error_string)

        # Set the list indices
        if simplified:
            range_start = 7
        else:
            range_start = 9

        # Filter the individuals
        header_list = header.split('\t')
        include_locations = []
        for sample_loc in range(range_start, len(header_list)):
            if sample_info[header_list[sample_loc]][0] in population_list:
                include_locations.append(sample_loc)

        # Open the output file
        with gzip.open(outfile, 'wb') as output:

            # Print header line
            header_line = header_list[0:range_start]
            for location in include_locations:
                header_line.append(header_list[location])

            output.write(''.join(['\t'.join(header_line), '\n']).encode('utf8'))

            # Loop through file
            while infile:
                line = infile.readline().decode('utf8').rstrip()

                if line:
                    fields = line.split('\t')
                else:
                    break

                # Include initial columns
                outstring = fields[0:range_start]

                # Include only individuals that pass the filter
                for location in include_locations:
                    outstring.append(fields[location])

                # Print final output
                output.write(''.join(['\t'.join(outstring), '\n']).encode('utf8'))
 
def _logme(output, logfile=sys.stderr, print_level=0):
    """Print a string to logfile
       From: https://raw2.github.com/MikeDacre/handy-functions/master/mike.py"""
    import datetime

    timestamp   = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
    output      = str(output)
    timeput     = ' | '.join([timestamp, output])

    stderr = False
    stdout = False

    try:
        if isinstance(logfile, str):
            with open(logfile, 'a') as outfile:
                print(timeput, file=outfile)
        elif getattr(logfile, 'name') == '<stderr>':
            print(timeput, file=logfile)
            stderr = True
        elif getattr(logfile, 'name') == '<stdout>':
            print(timeput, file=logfile)
            stdout = True
        elif getattr(logfile, 'mode') == 'a':
            if getattr(logfile, 'closed'):
                with open(logfile.name, 'a') as outfile:
                    print(timeput, file=outfile)
            else:
                print(timeput, file=logfile)
        else:
            logfile.close()
            with open(logfile, 'a') as outfile:
                print(timeput, file=outfile)
    except AttributeError:
        pass

    if print_level == 1 and not stdout:
        print(output)
    elif print_level == 2 and not stderr:
        print(output, file=sys.stderr)
 
def _get_args():
    """Command Line Argument Parsing"""
    import argparse

    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required Files
    parser.add_argument('infiles', nargs='+',
                        help="VCF file or files, gzip compressed")

    # Optional Files
    parser.add_argument('-p', '--panel_file',
                        help="Panel file for use in filtering, see example above")

    # Optional Arguments
    parser.add_argument('--filter_population',
                        help="Filter based on population. Uses panel_file. Provide comma-separated list of populations")
    parser.add_argument('-t', '--threads', nargs='?', default=default_threads,
                        help=''.join(["Threading, for running on multiple files. ",
                                      "Default: ", str(default_threads)]) )
    parser.add_argument('-o', '--output_directory', 
                        help="Directory to place output files. Default is current working directory")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('-l', '--logfile',
                        help="Optional log file for verbose output, Default STDERR (append mode)")

    return parser

# Main function for direct running
def main():
    """Run directly - use multithreading"""

    # Get commandline arguments
    parser = _get_args()
    args = parser.parse_args()

    # Run in filtering mode if filter is selected
    panel_file      = ''
    population_list = ''

    # Choose output Directory
    if args.output_directory:
        output_directory = path.realpath(args.output_directory)
    else:
        output_directory = path.realpath('.')

    if args.filter_population:
        # Make sure panel file exists
        if args.panel_file and path.isfile(args.panel_file):
            panel_file = path.realpath(args.panel_file)
            population_list = args.filter_population.split(',')
        else:
            parser.print_help()
            print("ERROR: filtering requires a panel file, make sure you have ",
                  "include a panel file with the '-p' flag and that that file exists",
                  file=sys.stderr)
            sys.exit(1)

        # Run filtering
        filter(args.infiles, population_list, panel_file, output_directory, args.threads, args.verbose, args.logfile)

    # Otherwise parse vcf file into simple version
    else:
        # Get threads
        pool = Pool(processes=int(args.threads))
        running_processes = []

        # Queue up the vcf_simplify instances
        for vcf_file in args.infiles:
            running_processes.append(pool.apply_async(vcf_simplify, (vcf_file, output_directory, args.logfile, args.verbose)) )

        # Run threads
        for process in running_processes:
            process.get()

    return

# The end
if __name__ == '__main__':
    main()
