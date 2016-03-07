"""
Compare all SNPs to the equivalent position in a genome file.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-15-13 18:01
 Last modified: 2016-03-01 14:24

   DESCRIPTION: Take a SNP file in either Bed or VCF format and create a
                list of Chromsome objects, which contain all of the SNPs.
                Then parse a list of FastA files and create a list of
                SeqIO objects.
                Finally, compare all SNPs to the equivalent position in the
                SeqIO objects, and create three lists to describe matches:
                    ref, alt, and no_match
                These lists are added to the Chromosome object and contain
                the positions of the SNPs that they describe.

          NOTE: All files can be plain text, gzipped, or bz2zipped

         USAGE: snps   = parse_[bed|vcf](snp_file)
                genome = create_seqio_list(list_of_fasta_files)
                comp_snps_to_chr(snps, genome)

============================================================================
"""
import os
import sys
import gzip
import bz2
import Bio.SeqIO
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["SNP", "Chromosome", "comp_snps_to_chr", "parse_bed",
           "parse_vcf", "create_seqio_list", "print_lists", "output_table"]


###############################################################################
#                                   Classes                                   #
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

    def get_lists(self, chromosome):
        """ Add three lists: ref, alt, and no_match
            Each will contain a list of positions that match
            either ref, alt, or nothing.

            chromosome must be a Bio.SeqIO object created from
            the same chromosome as this object
        """
        #  chr_id = chromosome.id
        chr_id = chromosome.id.rstrip('_maternal').rstrip('_paternal')
        chr_id = chr_id[3:] if chr_id.startswith('chr') else chr_id
        try:
            assert str(chr_id) == str(self.number)
        except AssertionError:
            raise AssertionError(("The provided chromosome does not have " +
                                  "the right ID. It's ID: {0}. Our ID: {1}\n" +
                                  "It's name: {2}. Our name: {3}\n").format(
                                      chr_id, self.number,
                                      chromosome.id, self.name))

        for pos, snp in self.snps.items():
            try:
                c = chromosome[pos].upper()
            except IndexError:
                sys.stderr.write(pos + '\n')
                raise
            if c == snp.ref.upper():
                self.ref.append(pos)
            elif c == snp.alt.upper():
                self.alt.append(pos)
            else:
                self.no_match.append(pos)

        self._lists_made = True

    def __init__(self, name):
        self.ref         = []
        self.alt         = []
        self.no_match    = []
        self._lists_made = False
        self.snps        = {}
        self.name        = name
        self.number      = name[3:] if name.startswith('chr') else name

    def __repr__(self):
        out_string = "Chromosome: {0} ({1})\n".format(self.name, self.number)
        out_string = out_string + '{0:>30}: {1}\n\n'.format(
            "Total SNP Count", len(self.snps))
        if self._lists_made:
            out_string = out_string + '{0:>30}: {1}\n'.format(
                'Ref alleles', len(self.ref))
            out_string = out_string + '{0:>30}: {1}\n'.format(
                'Alt alleles', len(self.alt))
            out_string = out_string + '{0:>30}: {1}\n\n'.format(
                'NO MATCH', len(self.no_match))
        else:
            out_string = out_string + "Allele lists not generated yet\n"
        return out_string

    def __len__(self):
        return len(self.snps)


###############################################################################
#                              Primary Functions                              #
###############################################################################


def comp_snps_to_chr(snps, genome):
    """ For every chromosome in genome loop through SNPs and run
        Chromsome.get_lists() on the corresponging Chromosome object
        snps must be a dictionary of Chromsome objects
        genome must be either a single SeqIO object or a list of SeqIO objects
        Note: The leading 'chr' is stripped from chromosome for comparison
    """
    ret_code = 0  # Return code for function

    # Create a name mapping from snp dictionary for lookup
    names = {}
    for nm in snps.keys():
        names[nm.lstrip('chr_')] = nm

    # Make genome into a list if isn't already
    if not isinstance(genome, list):
        genome = [genome]

    # Loop through chromosomes and run
    for seq_obj in genome:
        for chromosome in seq_obj:
            try:
                #  snps[names[chromosome.id.lstrip('chr_')]].get_lists(chromosome)
                snps[names[chromosome.id.lstrip('chr_').rstrip('_maternal').rstrip('_paternal')]].get_lists(chromosome)
            except KeyError:
                sys.stderr.write(('\nChromsome {0} is not in snp list, ' +
                                  'skipping.\nsnp list contains the ' +
                                  'following chromosomes: {1}.\n' +
                                  'Chromsome detail:\n{2}\n').format(
                                      chromosome.name,
                                      repr(names),
                                      chromosome))
                ret_code = 2

    return ret_code


def parse_bed(bed_file, base=0):
    """ Return a dictionary of Chromosome objects from a bed file
        File can be plain, gzipped, or bz2zipped
        'base' is subtracted from the position
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
            chromosomes[f[0]].add_snp(int(f[1]) - base, SNP(ref, alt))

    return chromosomes


def parse_vcf(vcf_file, base=1):
    """ Return a dictionary of Chromosome objects from a vcf file
        File can be plain, gzipped, or bz2zipped
        Return as base 0, i.e. subtract 'base' from the position
    """
    try:
        assert isinstance(base, int)
    except AssertionError:
        raise AssertionError('base must be an integer, is {}'.format(
            type(base)))

    chromosomes = {}

    with open_zipped(vcf_file) as infile:
        for line in infile:
            if line.startswith('#'):
                continue
            f = line.rstrip().split('\t')
            if f[0] not in chromosomes:
                chromosomes[f[0]] = Chromosome(f[0])
            chromosomes[f[0]].add_snp(int(f[1]) - base, SNP(f[3], f[4]))

    return chromosomes


def create_seqio_list(chromosome_list):
    """ Take a list of fasta files (one is fine) and create a single
        SeqIO object for the whole genome
    """
    unsorted_dict = {}
    for file in chromosome_list:
        infile = open_zipped(file)
        name = os.path.basename(file).split('.')[0]
        unsorted_dict[name] = Bio.SeqIO.parse(infile, 'fasta')
    return [i[1] for i in sorted(unsorted_dict.items(),
                                 key=chr_comp)]


###############################################################################
#                              Display Functions                              #
###############################################################################


def print_lists(chr_list):
    """ Take a list of Chromosome objects and print their counts to STDOUT
        chr_list can also be a single Chromosome object or a dictionary
        NOTE: ref, alt, and no_match should already be populated
    """
    chr_list = _clean_chr_list(chr_list)
    for chrm in chr_list:
        # Print the lists
        sys.stdout.write('{0}\n'.format(chrm))


def output_table(chr_list, outfile=sys.stdout):
    """ Take a list of Chromosome objects and write their counts to a table
        chr_list can also be a single Chromosome object or a dictionary
        NOTE: ref, alt, and no_match should already be populated
    """
    chr_list = _clean_chr_list(chr_list)
    with open_zipped(outfile, 'w') as fout:
        fout.write('Chr\tRef_Count\tAlt_Count\tNo_Match\n')
        for chrm in chr_list:
            fout.write('{}\t{}\t{}\t{}\n'.format(chrm.name, chrm.ref,
                                                 chrm.alt, chrm.no_match))


def plot_snps(chr_list):
    """ Take a list of Chromosome objects and plot their counts as a bar chart
        chr_list can also be a single Chromosome object or a dictionary
        NOTE: ref, alt, and no_match should already be populated
    """
    plot_snps = _clean_chr_list(chr_list)
    groups = np.arange(len(chr_list))
    width = 0.35

    plt.close()
    fig, ax = plt.subplots()

    # Plot the bars
    refbars = ax.bar(groups,
                     [len(i.ref) for i in plot_snps], width, color='g')
    altbars = ax.bar(groups + width,
                     [len(i.alt) for i in plot_snps], width, color='b')
    nmtbars = ax.bar(groups + (width * 2),
                     [len(i.no_match) for i in plot_snps], width, color='r')

    # Format the graph
    ax.set_ylabel('Counts')
    ax.set_title('Counts per Chromosome')
    ax.set_xticks(groups + width + width)
    ax.set_xticklabels([i.name for i in plot_snps], rotation=-90)

    ax.legend((refbars[0], altbars[0], nmtbars[0]),
              ('Ref', 'Alt', 'No Match'))

    ax.autoscale(enable=True, axis='x', tight=True)

    return fig


###############################################################################
#                              Private functions                              #
###############################################################################


def chr_comp(keys):
    """ Allow numeric sorting of chromosomes by chromosome number
        If numeric interpretation fails, position that record at -1
    """
    if isinstance(keys, (list, tuple)):
        key = keys[0]
    elif isinstance(keys, compare_snps_to_genome.Chromosome):
        key = keys.number
    else:
        key = keys
    key = key.lower().replace("_", "")
    chr_num = key[3:] if key.startswith("chr") else key
    if chr_num == 'x':
        chr_num = 98
    elif chr_num == 'y':
        chr_num = 99
    elif chr_num.startswith('m'):
        chr_num = 100
    else:
        try:
            chr_num = int(chr_num)
        except ValueError:
            chr_num = 101
    return chr_num


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


def _clean_chr_list(chr_list):
    """ Take a single, list, dictionary, or tuple of Chromosomes
        and return an iterable list
    """
    if isinstance(chr_list, dict):
        chr_list = [i[1] for i in sorted(chr_list.items(), key=chr_comp)]
    if not isinstance(chr_list, (list, tuple)):
        chr_list = list(sorted(chr_list, key=chr_comp))
    return chr_list

# vim:set et sw=4 ts=4 tw=79:
