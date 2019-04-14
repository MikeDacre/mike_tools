#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert the output of CombineSnpCounts.py to gene level counts

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       CREATED: 2016-32-20 13:05
 Last modified: 2016-05-27 16:05

   DESCRIPTION: Takes output, pre-filtered by FDR, and returns gene-level
                data where the pvalue is the median of the pvalues of the
                child SNPs.

============================================================================
"""
import os
import sys
import argparse
import gzip
import bz2
import random
from collections import defaultdict

# Progress bar
from subprocess import check_output
from tqdm import tqdm

# Handle python objects
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Math and calculations
import math
import bisect
import numpy
import scipy.stats
import pandas

import logme

#################################################
#  Classes to hold exon information for lookup  #
#################################################


class Chrom(list):

    """A list of genes on one chromosome."""

    def find(self, loc, strand=None):
        """Search in every gene range."""
        loc = int(loc)
        for exon in self:
            if exon.find(loc, strand):
                return exon.gene
        return None

    def __repr__(self):
        """Print a list of genes."""
        astr = []
        for gene in self:
            astr += [repr(gene)]
        return '; '.join(astr)


class Exon(object):

    """A single exon, gene points to a Gene object."""

    def __init__(self, gene, start, end, strand):
        """Create an Exon, gene must be an existing Gene object."""
        self.start  = int(start)
        self.end    = int(end)
        self.strand = strand

        assert isinstance(gene, Gene)
        self.gene   = gene

        # Set a back link to us
        self.gene.exons.append(self)

    def find(self, loc, strand=None):
        """Return True if loc in self. If strand provided, check that."""
        if strand and not strand == self.strand:
            return False
        return self.start <= loc < self.end

    def __repr__(self):
        return "Exon<{}({}:{})>".format(self.gene.name, self.start, self.end)


##########################################
#  Class to hold Genes and their counts  #
##########################################


class Gene(object):

    """A single gene, holds SNPs."""

    def __init__(self, name, trans_id=''):
        """Create an empty Gene object."""
        self.name     = name
        self.trans_id = trans_id

        # All the SNPs in this Gene
        self.snps         = []

        # All the exons
        self.exons        = []

        # Raw counts
        self.mat_counts   = 0
        self.pat_counts   = 0
        self.other_counts = 0

        # Winners
        self.mat_win = 0
        self.pat_win = 0
        self.has_ase = 0
        self.no_ase  = 0
        self.weird   = 0
        self.failed  = 0

        # Significant metrics for all SNPs in this Gene
        self.pval = None
        self.win  = None

    def sum_counts(self):
        """Add up all counts in snps."""
        self.mat_counts   = 0
        self.pat_counts   = 0
        self.other_counts = 0
        for snp in self.snps:
            self.mat_counts   += snp.mat_counts
            self.pat_counts   += snp.pat_counts
            self.other_counts += snp.other_counts

    def calc_pval(self):
        """Take the median of the pvalues of all SNPs as our pvalue.

        We only include SNPs with no Ns or non-parental alleles.
        """
        pvals = []
        for snp in self.snps:
            pvals.append(snp.pval)
        pvals = [p for p in pvals if isinstance(p, float)]
        if pvals:
            self.pval = numpy.median(pvals)
        else:
            self.pval = numpy.nan

    def calc_winner(self):
        """Sum winners, try to pick Gene-level winner.

        Only chooses Gene-level winner if all SNPs aggree.

        Sets self.win to one of:
            'mat', 'pat', 'WEIRD', '?', or 'NS' (for non-significant)

        Ratio calculations use total SNP counts, not the sum of the parental
        alleles.
        """
        for snp in self.snps:
            if not hasattr(snp, 'win') or not snp.win:
                snp.calc_winner()
            if not snp.win:
                continue
            if snp.win == 'M':
                self.mat_win += 1
            elif snp.win == 'P':
                self.pat_win += 1
            elif snp.win == '?':
                self.failed += 1  # Not bothering with greater res now.
            if snp.cls == 'Sym':
                self.no_ase += 1
            elif snp.cls == 'Asym':
                self.has_ase += 1
            elif snp.cls == 'Weird':
                self.weird += 1

        # Winner must have more than 60% of alleles with gene-level
        # significance
        if not self.pval:
            self.calc_pval()
        if not self.pval:
            logme.log('No pvalue for gene {}'.format(self), 'debug')
            self.win = 'NA'
            return

        if self.weird/len(self) > 0.4:
            self.win = 'WEIRD'
        elif self.mat_win > self.pat_win and self.mat_win/len(self) > 0.6:
            self.win = 'mat'
        elif self.pat_win > self.mat_win and self.pat_win/len(self) > 0.6:
            self.win = 'pat'
        else:
            self.win = '?'

    def __len__(self):
        """How many SNPs are in this gene."""
        return len(self.snps)

    def __repr__(self):
        """Summary info."""
        self.sum_counts()
        return "{}(mat:{};pat:{};other:{})".format(self.name,
                                                   self.mat_counts,
                                                   self.pat_counts,
                                                   self.other_counts)

###############################
#  Class to hold SNP alleles  #
###############################


class SNP(object):

    """A SNP.

    Contains::
        chrm        -- chromosome
        pos         -- position
        mat_allele  -- The maternal allele
        pat_allele  -- The paternal allele
        counts      -- Dict of raw counts for each base indexed by ATCG
        win         -- M/P/?
        cls         -- Sym/Asym/Weird -- Asym: ASE
        pval        -- binomial pvalue
        gene        -- the parent Gene class
        mat_counts
        pat_counts
        other_counts

    """

    def __init__(self, gene, snpinfo):
        """Create a SNP object.

        :gene:    A gene name refering to a Gene object
        :snpinfo: A tuple from alleleseq

        """
        self.gene = gene
        self.gene.snps.append(self)

        # Get info
        (self.chrm, self.pos, ref, mat_gtyp, pat_gtyp, c_gtyp, phase,
         self.mat_allele, self.pat_allele, cA, cC, cG, cT,
         self.win, self.cls, pval, BindingSite,
         cnv) = snpinfo
        self.pval = float(pval)

        # Set counts
        self.counts = {'A': int(cA), 'C': int(cC), 'G': int(cG), 'T': int(cT)}

        # Assign maternal/paternal
        self.mat_counts = self.counts[self.mat_allele]
        self.pat_counts = self.counts[self.pat_allele]

        # Count others
        count = list(self.counts)
        count.remove(self.mat_allele)
        count.remove(self.pat_allele)
        self.other_counts = self.counts[count.pop()] + self.counts[count.pop()]

    def __len__(self):
        """The total number of SNPs."""
        return numpy.sum(self.counts.values())

    def __repr__(self):
        """Summary info."""
        return "SNP<(mat:{};pat:{};other:{})>".format(
            self.mat_counts, self.pat_counts,
            self.other_counts)


###############################################################################
#                             Parse the Bed File                              #
###############################################################################


def parse_gene_bed(bed_file):
    """Return a defaultdict of Chrom objects for lookup.

    To lookup, just run exons[chromsome].find(location) (where exons is the
    defaultdict returned by this function).

    NOTE: Uses entire gene, not exons. That was more useful for this
          application.

    :returns: defaultdict(exons), dict(genes)
    """
    # Initialize the list exons for lookup
    exons = defaultdict(Chrom)

    # Initialize a dictionary of genes
    genes = {}

    count = 0
    with open_zipped(bed_file) as fin:
        for line in fin:
            count += 1
            if line.startswith('#'):
                continue
            fields = line.rstrip().split('\t')
            chrom  = chr2num(fields[0])
            try:
                start  = int(fields[1])+1  # Enforce 1-base, bed is 0-based
            except IndexError:
                print(count)
            end    = int(fields[2])+1
            gene   = fields[3]
            trans  = gene
            strand = fields[5]
            if gene not in genes:
                genes[gene] = Gene(gene, trans)
            # Assign exons
            starts  = [start+int(i) for i in fields[11].rstrip(',').split(',')]
            lengths = [int(i) for i in fields[10].rstrip(',').split(',')]
            assert len(starts) == len(lengths)
            assert len(starts) == int(fields[9])
            for strt in starts:
                exon = Exon(genes[gene], strt,
                            strt+lengths[starts.index(strt)], strand)
                exons[chrom].append(exon)

    return exons, genes

###############################################################################
#                                Main Function                                #
###############################################################################


def get_gene_counts(bed_file, alleleseq_output, chrom_to_num=False,
                    logfile=sys.stderr):
    """Return a list of Gene objects from all snps in exons.

    :chrom_to_num: If true, convert 'chr1' to 1
    """
    logme.log('Parsing gene bed')
    exons, genes = parse_gene_bed(bed_file)

    # Stats
    total_count = 0
    not_in_gene = 0

    snps = []

    # Parse the file
    logme.log('Parsing alleleseq output')
    lines = int(check_output(['wc', '-l', alleleseq_output]).decode().split(' ')[0])
    with open_zipped(alleleseq_output) as fin:
        # File format test
        header = fin.readline()
        if not header.startswith('chrm'):
            raise Exception("Invalid alleleseq file format")
        # Loop through the file
        siter = tqdm(fin, unit='snps', total=lines) if 'PS1' in os.environ \
                else fin
        for line in siter:
            snpinfo = line.rstrip().split('\t')
            total_count += 1
            chrm = chr2num(snpinfo[0]) if chrom_to_num else snpinfo[0]
            gene = exons[chrm].find(int(snpinfo[1]))
            # Skip everything not in a gene
            if gene is not None:
                # The SNP adds itself to the genes list
                s = SNP(gene, snpinfo)
                snps.append(s)
            else:
                not_in_gene += 1

    newgenes = {}
    for name, gene in genes.items():
        if gene:
            gene.sum_counts()
            gene.calc_pval()
            gene.calc_winner()
            newgenes[name] = gene
    return newgenes


###############################################################################
#             AlleleSeq's Bionomial Test Functions from binom.py              #
###############################################################################


def binomtest(x, n, p):
    """Run a binomial test with scipy unless n*p>50, then use normal_approx."""
    #return (scipy.stats.binom_test(x, n, p), normal_approx(x, n, p))
    if n*p > 50:
        return normal_approx(x, n, p)
    else:
        return scipy.stats.binom_test(x, n, p)


def normal_approx(x, n, p):
    """A different implementation of the binomial test?."""
    if abs(x-n*p) < 1e-5:
        return 1.0
    u=p*n
    s=math.sqrt(n*p*(1-p))
    norm=scipy.stats.distributions.norm(u,s)
    if x<n*p:
        pval=2*norm.cdf(x+.5) # add 0.5 for continuity correction
    else:
        pval=2*(1-norm.cdf(x-.5))
    return pval



###############################################################################
#                 Calc FDR for genes use AlleleSeq Algorithm                  #
###############################################################################


class binomMemo(object):

    """Do a binomial test with a definied range."""

    def __init__(self, n):
        """Create a binomial range."""
        self.n=n
        self.cache=[[binomtest(j, i, 0.5) for j in range(i+1)] for i in range(n)]
    def binomtest(self, a, cnt):
        """Do a binomial test."""
        if cnt<self.n:
            return self.cache[cnt][a]
        else:
            return binomtest(a, cnt, 0.5)


def simpval(cnt,bm):
    """Simulate a binomial pvalue from cnt."""
    a=sum([random.randint(0,1) for i in range(cnt)])
    pval=bm.binomtest(a, cnt)
    return pval


def simpval2(cnt,bm):
    """Simulate a binomial pvalue from cnt."""
    a=sum([random.randint(0,1) for i in range(cnt)])
    #  pval=bm.binomtest(a, cnt)
    return a



def calc_fdr(pvals, target=0.1, sims=5, verbose=False):
    """Return the highest pvalue that beats an FDR of 'target'.

    I have kept most of the bloat from the original algorithm, and only removed
    lines that had to be removed, all of these were just file handling lines.

    :pvals:   A tuple of (mat_count, pat_count, p-value). Used to simulate new
              pvalue set.
    :target:  The FDR cutoff to beat.
    :sims:    The number of simulations to do when calulating the random set of
              pvalues.
    :verbose: Print extra information.
    :returns: The pvalue that beats the FDR target for this count set.
    """
    bestFDR=bestPV=None

    random.seed(0)  # This is set in the original algorithm

    #  print "#"," ".join(sys.argv)
    #  print "pval\tP\tFP\tFDR"
    bm=binomMemo(60)
    #  h=getInterestingHetsAnnotations.Handler(ifile, hasHeader=True)
    #  n=h.getCount()  # Returns the number of lines in the file
    #  g=h.getAllAnnotationsGenerator();  # All this returns is the infile as {chr => {pos => rest of the file as a tuple}}

    n = len(pvals)

    act_pvals=numpy.zeros(n) # pval as reported in counts file
    cnt_sums=numpy.zeros(n, dtype=numpy.int)  # sum of major and minor alleles

    # for each hetSNP, get the count of major and minor allele from the input file
    for i, t in enumerate(pvals):
        mat, pat, pval = t
        act_pvals[i] = float(pval)  # Append p-value to the array
        counts = [mat, pat]  # Create a list of counts
        counts = [int(e) for e in counts] # Make them integers
        counts = sorted(counts, reverse=True)[0:2] # Keep only the top two
        cnt_sums[i] = sum(counts)  # Sum the top two counts

    act_pvals = sorted(act_pvals)
    # For every line in the input file, calculate a random pvalue. Repeat this
    # sims times. Sims is often 5.
    sim_pvals=numpy.array([ sorted([simpval(cnt_sums[j],bm) for j in range(n)]) for i in range(sims)])
    #sim_pvals_means=numpy.mean(sim_pvals, 0)

    pvs=[e*0.001 for e in range(10)]+[e*0.01 for e in range(1,10)]+[e*0.1 for e in range(1,10)]
    # for a given test pv, find the number of actual pvals that are smaller, and the number of sim pvals that are smaller.
    # FDR is the ratio
    sys.stderr.write("pval\tpos_in_actual\tmean_sim_pos\tFDR\n")
    for pv in pvs:
        # Get what position the pvalue from pvs is in in the actual pvalues
        # from the input file.
        Nact=bisect.bisect(act_pvals, pv)
        # For every simulated pvalue set, find the position of the pvalue from
        # pvs in that set, then take the mean of all simulations.
        mean_Nsims=numpy.mean([bisect.bisect(sim_pvals[i], pv) for i in range(sims)])
        # The false discovery rate is the position of the pvalue from pvs in
        # the simulated pvalue set divided by the position of the same pvalue
        # in the actual pvalue set from the infile, plus 1.
        FDR=mean_Nsims/(Nact+1)
        sys.stderr.write("%f\t%s\t%f\t%f\n" % (pv, Nact, mean_Nsims, FDR))

    # This is my attempt to find the act_pval that corresponds best to the desired target FDR.
    # This version walks from largest observed pvalue to the smallest.
    if target:
        last_FDR=last_pv=0.0
        for Nact, pv in sorted(enumerate(act_pvals), reverse=True):
            # For every simulated pvalue set, find the position of the pvalue from
            # the actual pvalues in the simulated pvalues, then take the mean
            # of all simulations.
            mean_Nsims=numpy.mean([bisect.bisect(sim_pvals[i], pv) for i in range(sims)])
            # The false discovery rate is the position of the pvalue from the
            # actual data in the simulated pvalue set divided by the position
            # we are in the list of pvalues (walking from largest to smallest)
            FDR=mean_Nsims/(Nact+1)
            if verbose:
                sys.stderr.write("test %d %f %f %f\n" % (
                    Nact,mean_Nsims,FDR, pv))
            # As soon as we get an FDR that is less than the target (usually
            # 0.1), that is our 'bestFDR': the largest p-value that beats our
            # target FDR.
            if not bestFDR and FDR < target:
                sys.stderr.write("target %f\n" % target)
                sys.stderr.write("before %f %f\n" % (last_FDR, last_pv))
                sys.stderr.write("after  %f %f\n" % (FDR, pv))
                bestFDR = FDR; bestPV = pv

            last_FDR=FDR; last_pv=pv

        sys.stderr.write("Target {} FDR {} pv {}\n".format(target,
                                                           bestFDR,
                                                           bestPV))

        return bestFDR


###############################################################################
#                     Create pandas dataframe from genes                      #
###############################################################################


def genes_to_df(genes, ind):
    """Make a pandas dataframe from a dictionary of genes.

    Datafram has the following columns::
        Counts::
            'Mat_Counts'  -- Total number of maternal counts for this gene
            'Pat_Counts'  -- Total number of paternal counts for this gene
            'N_Counts'    -- Total number of reads with N in the SNP position
            'Others'      -- Total number of reads with a non-parental allele
        Gene-level summary::
            'Winner'      -- The overall winner ('mat' or 'pat')
            'pval'        -- The pvalue of that association (binomial)
        SNP-level information::
            'SNPs'        -- Total number of SNPs in this gene
            'Mat_wins'    -- Total number of SNPs with materal wins
            'Pat_wins'    -- Total number of SNPs with pateral wins
            'Not_Sig'     -- Total number of SNPs that weren't significant
            'Weird'       -- Total number of SNPs with non-parental allele
            'Failed'      -- Total number of SNPs that failed for some reason,
                             (usually due to Ns in the sequence)
    """
    ind = str(ind)
    # Populate dictionaries from every gene
    df_dict = {}
    if not genes:
        raise Exception('Genes must have at least one entry.')
    for name, gene in genes.items():
        gene.sum_counts()
        df_dict[ind + '_' + name] = {'Mat_Counts': gene.mat_counts,
                                     'Pat_Counts': gene.pat_counts,
                                     'Others':     gene.other_counts,
                                     'SNPs':       len(gene),
                                     'Winner':     gene.win,
                                     'pval':       gene.pval,
                                     'Mat_wins':   gene.mat_win,
                                     'Pat_wins':   gene.pat_win,
                                     'Weird':      gene.weird,
                                     'Failed':     gene.failed,
                                     'TX':         name,
                                     'Tissue ID':  ind}
    column_order = ['TX', 'Tissue ID', 'Mat_Counts', 'Pat_Counts', 'Others',
                    'SNPs', 'Winner', 'pval', 'Mat_wins', 'Pat_wins',
                    'Weird', 'Failed']
    df = pandas.DataFrame.from_dict(df_dict, orient='index')
    df.index.name = 'IDX'
    df = df[column_order]

    return df


def chr2num(chrom):
    """Make chr# #."""
    return chrom[3:] if chrom.startswith('chr') else chrom


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


def main(argv=None):
    """Run as a script."""
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('exon_positions_bed',
                        help="A bed file of exons to include.")
    parser.add_argument('alleleseq_output',
                        help="The output of CombineSnpCounts.py, filtered " +
                        "FDR")

    parser.add_argument('-i', '--ind', help='Individual name')
    parser.add_argument('-n', '--tonum', action='store_true',
                        help='Convert chr# to #')

    # Optional Files
    optfiles = parser.add_argument_group('Optional Files')
    optfiles.add_argument('-o', '--outfile', default=sys.stdout,
                          help="Output file, Default STDOUT")
    optfiles.add_argument('-l', '--logfile', default=sys.stderr,
                          help="Log File, Default STDERR (append mode)")
    optfiles.add_argument('--data',
                          help="Output raw gene dictionary to this file.")
    optfiles.add_argument('--pandas',
                          help="Output a pickled pandas dataframe here.")

    # FDR Calulations
    fdrcalc = parser.add_argument_group('FDR Calculation')
    fdrcalc.add_argument('--filter-fdr', action='store_true',
                         help="Filter the output by FDR")
    fdrcalc.add_argument('-f', '--fdr-cutoff', type=float, default=0.1,
                         metavar='', help="FDR cutoff (Default 0.1).")
    fdrcalc.add_argument('-s', '--simulations', type=int, default=10,
                         metavar='',
                         help="# simulations for FDR calculation " +
                         "(Default: 10)")

    args = parser.parse_args(argv)

    ind = args.ind if args.ind \
            else os.path.basename(args.alleleseq_output).split('.')[0]

    genes = get_gene_counts(args.exon_positions_bed, args.alleleseq_output,
                            args.tonum)

    giter = tqdm(genes.values(), unit='genes') if 'PS1' in os.environ \
            else genes.values()
    for gene in giter:
        gene.sum_counts()

    if args.data:
        with open(args.data, 'wb') as fout:
            pickle.dump(genes, fout)

    df    = genes_to_df(genes, ind)

    fdr_pval = calc_fdr(
        [tuple(x) for x in df[['Mat_Counts', 'Pat_Counts', 'pval']].values],
        target=args.fdr_cutoff, sims=args.simulations)

    logme.log('In ind {} p-values smaller than {} beat FDR of {}'
              .format(ind, fdr_pval, args.fdr_cutoff), 'info')

    # Filter by FDR if requested
    if args.filter_fdr:
        logme.log('Filtering genes by FDR less than {}'
                  .format(args.fdr_cutoff), 'info')
        df = df[df.pval < fdr_pval]

    if args.pandas:
        df.to_pickle(args.pandas)

    with open_zipped(args.outfile, 'w') as fout:
        df.to_csv(fout, sep='\t')

    return 0

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
