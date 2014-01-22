#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Copyright © Mike Dacre <mike.dacre@gmail.com>
#
# Distributed under terms of the MIT license
"""
====================================================================================

          FILE: vcf_simplify (python 3)
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, Property of Stanford, Use however you wish
       VERSION: 0.1
       CREATED: 2014-01-21 17:38
 Last modified: 2014-01-22 13:07

   DESCRIPTION: Take a compressed vcf file such as 
                ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/ALL.chr1.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz
                and create a simplified matrix where all genotypes are represented 
                as 0/1/2 where 0: homozygote_1; 1:heterozygote; 2: homozygote_2.

                Out file format (tab delimited):
                SNP_ID(e.g.rs58108140)\tCHR(e.g.1/MT)\tPOS(e.g.10583)\tref\talt\tqual\tfilter\t[sample_1]\t[sample_2]\t...\t[sample_n]

  REQUIREMENTS: 1. A 1000genomes-style VCF file with GT:DS:GL style genotypes (see note)
                2. A 1000genomes-style panel file 
                    ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/integrated_call_samples.20101123.ALL.panel

          NOTE: The vcf files must be gzipped, and the genotypes must be encoded as 
                0|0, 0|1, 1|1

 USAGE EXAMPLE: ./vcf_simplify.py -p integrated_call_samples.20101123.ALL.panel\
                ALL.chr1.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz\
                ALL.chr2.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz

       OUTPUTS: Files by chromosome and population, e.g.:
                CEU_chr1.txt.gz, CEU_chr2.txt.gz, YRI_chr1.txt.gz, YRI_chr2.txt.gz

====================================================================================
"""
import sys, re

def vcf_simplify(vcf_file, outfile='', logfile=sys.stderr, verbose=False):
    """Take a 1000genomes style vcf file (MUST BE GZIPPED) and 
       simplify it to:
       
       rsID\\tchr\\tpos\\tref\\talt\\tqual\\tfilter\\tsample_1\\t[sample_2]\\t...[sample_n]\\n"""
    import gzip

    # Get an outfile name:
    if not outfile:
        outfile = re.sub('.vcf.gz$','', vcf_file) + '_simplified.vcf.gz'

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

            while (infile):
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
                
def _logme(output, logfile=sys.stderr, print_level=0):
    """Print a string to logfile
       From: https://raw2.github.com/MikeDacre/handy-functions/master/mike.py"""
    import datetime

    timestamp   = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
    output      = str(output)
    timeput     = ' | '.join([timestamp, output])

    stderr = False
    stdout = False

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

    if print_level == 1 and not stdout:
        print(output)
    elif print_level == 2 and not stderr:
        print(output, file=sys.stderr)
                
def _get_args():
    """Command Line Argument Parsing"""
    import argparse, sys
    
    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Required Files
    parser.add_argument('infiles', nargs='+', 
                        help="VCF file or files, gzip compressed")
    
    # Optional Files
    parser.add_argument('-p', '--panel_file', 
                        help="Panel file, see example above")
     
    # Optional Arguments
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")
    parser.add_argument('-l', '--logfile', nargs='?', default=sys.stderr, 
                        type=argparse.FileType('a'),
                        help="Optional log file for verbose output, Default STDERR (append mode)")
    
    return parser

# Main function for direct running
def main():
    """Run directly"""
    # Get commandline arguments
    parser = _get_args()
    args = parser.parse_args()

    # Get sample info from panel file
    if args.panel_file:
        sample_info = parse_panel_file(args.panel_file, logfile=args.logfile, verbose=args.verbose)    

    # Parse vcf file into simple version
    for vcf_file in args.infiles:
        vcf_simplify(vcf_file, verbose=args.verbose, logfile=args.logfile)

# The end
if __name__ == '__main__':
    main()
