#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combine AlleleSeq counts across genes.

===============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-03-25 12:03
 Last modified: 2016-05-13 23:22

          IDEA: AlleleSeq counts alleles and assigns them to mat or pat
                and then counts alleles in all individuals for each SNP.
                However, what we want is the counts (maternal or paternal)
                for all SNPs in a gene. To replicate AlleleSeq's functionality
                we want to do the exact same binomial test for p-value for
                the gene level counts as AlleleSeq's CombineSnpCounts.py
                script does for SNP level counts.

                The one big difference is that the original AlleleSeq
                algorithm was able to assume that every SNP could be treated
                independently. Because we are looking in genes, we cannot do
                that; a very high proportion of SNPs in a single gene will
                be covered by a single paired end read set, meaning that
                neighboring SNP counts are not independent.

                To address this, we run the AllleleSeq binomial test per SNP,
                and then take the median of all p-values in a given gene as
                the p-value for that gene. We then try to estimate which
                allele won per SNP, and if then we ask if any one allele takes
                more than 60% of the wins for a gene. If that is the case and
                the gene level p-value beats 0.05, we say that the gene is
                biased towards a single parent.

                We also want to be able to filter the results by FDR, using
                the exact same algorithm used in AlleleSeq. To do that, we
                take the contents of FalsePos.py and modify them the work
                on our data, essentially we just iterate over genes instead
                of over SNPs.

                In addition, we want to be able to add additional gene level
                information to the final output.

   DESCRIPTION: This script has two primary modes, create and analyze. This is
                the case because building the SNP->Gene+Phased Allele map takes
                time (about 30 minutes for mouse data). So we build it first
                with create mode and then use it with analyze mode.

                In addition there is a flip mode, which will take all the SNPs
                in the pickled object from create mode and flip the maternal
                and parternal SNPs. This is useful if you have reciprocal
                hybrid data.

                A GTF or bed of gene locations must also be provided, and will
                be used to sum counts across genes. Note: there is a quirk in
                this pipeline due to what I needed to do::
                    If the file is GTF, only exons are used
                    If the file is bed, the whole gene length is used
                This is a silly way to do it, but it allows two ways of testing
                ASE in genes: one in only coding sequence, and one in the whole
                transcript length.

                Output is handled by pandas.

          NOTE: Currently only works on PHASED snps.

         USAGE: Three modes::
                    create:  Make a SNP object that can be used for analysis
                    analyze: Run the alogithm described above on counts files.
                    flip:    Flip mat and pat in the dictionary output by
                             create.

===============================================================================
"""
# Command line parsing and printing
import os
import sys
import argparse
import gzip
import bz2
from time import sleep
from textwrap import dedent

# File parsing
import re
from collections import defaultdict

# Handle python objects
try:
    import cPickle as pickle
except ImportError:
    import pickle

# Copy dictionaries
from copy import copy
from copy import deepcopy

# Math and calculations
import math
import bisect
import random
import numpy
import scipy.stats
import pandas

# Allow multiprocessing
from subprocess import check_call, call
from multiprocessing import Pool, cpu_count

# Logging functions
try:
    import logme
except ImportError:
    raise Exception(
        'You need the logme library to run this script, you ' +
        'can get it here: ' +
        'https://github.com/MikeDacre/mike_tools/blob/master/python/logme.py')

logme.LOGFILE   = sys.stderr
logme.MIN_LEVEL = 'info'

JOBS = cpu_count()

SNP_NULL = 0
GENE_NULL = 0

# Make it easy to import all classes. This allows you to work with the
# pickled data by::
# import alleleseq_genes
# from alleleseq_gene import *
__all__ = ['Chrom', 'Exon', 'Gene', 'SNP', 'SNPFile']

# If the ratio all counts in either the mat or pat allele to counts
# in another allele is below this threshold report as 'WEIRD'
THRESH1 = 0.9

# p-value threshold to call winner
THRESH2 = 0.05

# If the ratio of N counts is above this threshold, discard
NTHRESH = 0.25

# Reset broken multithreading
call("taskset -p 0xff %d &>/dev/null" % os.getpid(), shell=True)

# Original notes in FalsePos.py
FDR_INFO = ''' Some notes on what is going on here.
Basically, we want to use simulation to explicitly calculate a FDR for binomial tests on unbalanced alleles.  We use
a binomial pvalue test to determine whether the ratio of alleles departs significantly from what would be expected
from a fair coin toss.

However, since the number of trials varies a lot from one test to another, it seems best to use an explicit method.

Imagine that you want a particular overall FDR, say 0.1, for the entire dataset.  Then, what pvalue threshhold would correspond to that?

say we have n trials, where the number of coin flips in a trial varies, and is given by cnt(i)

FDR = Nsim/Nact, where:

Nsim = sum( indicator(test(i) < pval)) over i.  This is the number of trials of the fair coin that had a "surprising" outcome, i.e.
were further in the tail than the pval threshold.  In a perfect, non-discrete world, Nsim/n would equal pval, but the whole point of this
exercise is that in the discrete, imperfect world it doesnt.

Nact = the number of actual datapoints observed to have a binomial probability less than pval.

So, for a given pval, say 0.05, we can calculate the FDR, which will be larger.  The first output from this code consists of a nice sampling of
example pvals and their corresponding FDR.  We are interested in the reverse of this, i.e. having picked an FDR, we want the pval that would best give us
this FDR.

Thats the point of the second part of the output.  Starting from the largest pval, we work our way down, calculating FDR for each test,
until FDR falls below our target.

Note that FDR is NOT monotonically decreasing as we do this. Its true that both Nsim and Nact decrease.  However, Nact is strictly decreasing, but Nsim can hold steady, which results in temporarily increasing FDR over that interval.

Also note that we do multiple simulations and use the mean of the Nsim values, in order to smooth out the results.

'''

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
#                  AlleleSeq's p-value Calculation Function                   #
###############################################################################


def testCounts(counts, chrm, snprec):
    """Assign a winning parent and pvalue to a single set of counts.

    This function is not used in this script but is here for reference.
    calc_pval is supposed to do **EXACTLY** the same thing, but using the
    Gene object syntax I have created here.

    The bulk of this function is replicated in the SNP class methods::
        SNP.calc_pval()
        SNP.calc_winner()

    To do the same thing for genes, the identically named methods in the Gene
    object take the median p-value for all SNPs in the gene, and try to call
    the gene-level winner from SNP level winners.
    """
    winningParent='?'
    ref_pos, mat_genotype, pat_genotype, child_genotype, mat_allele, pat_allele, typ, ref, hetSNP = snprec

    # first, make sure that the expected alleles are the bulk of the counts
    total = counts['a']+counts['c']+counts['g']+counts['t']
    # I *think* a1 is the maternal genotype and a2 is the paternal
    a1,a2=convert(child_genotype)
    if a1==a2:
        # Homozygote
        allelecnts = counts[a1]
    else:
        # Heterozygote
        allelecnts = counts[a1]+counts[a2]

    # Both mat an pat counts summed
    both=counts[a1]+counts[a2]

    # This ranks the counts from largest to smallest
    sortedCounts=sorted([(counts['a'], 'a'), (counts['c'],'c'), (counts['g'], 'g'), (counts['t'], 't')], reverse=True)
    # The largest number of counts
    majorAllele=sortedCounts[0][1]

    # The smallest of the mat/pat counts
    smaller=min(counts[a1], counts[a2])
    #pval=binomialDist.cdf(smaller, both, 0.5)*2 # This had problems for large sample sizes.  Switched to using scipy
    pval = binomtest(smaller, both, 0.5) # scipy.binom_test was unstable for large counts

    if float(allelecnts)/total < THRESH1:
        print >>LOGFP,  "WARNING %s:%d failed thresh 1 %d %d" % (chrm, ref_pos, allelecnts, total)
        return (WEIRD, pval, a1, a2, counts, winningParent)

    # if the snp was phased
    if mat_allele and pat_allele:
        if mat_allele.lower()==majorAllele.lower():
            winningParent='M'
        elif pat_allele.lower()==majorAllele.lower():
            winningParent='P'
        else:
            winningParent='?'

    if a1!=a2:
        # we expect roughly 50/50.
        if pval < THRESH2:
            print >>LOGFP,  "NOTE %s:%d Looks interesting: failed thresh 2 %d %d %f" % (chrm, ref_pos, both, smaller, pval)
            print >>LOGFP,  "SNPS %s/%s, COUNTS a:%d c:%d g:%d t:%d" % (a1, a2, counts['a'], counts['c'], counts['g'], counts['t'])
            print >>LOGFP,  "Phasing P:%s M:%s D:%s" % (pat_allele, mat_allele, snprec)
            print >>LOGFP,  "\n"
            return (ASYMMETRIC, pval, a1, a2, counts, winningParent)
        else:
            return (SYMMETRIC, pval, a1, a2, counts, winningParent)
    else:
        return (HOMOZYGOUS, pval, a1, a2, counts, winningParent)


###############################################################################
#                AlleleSeq's FDR calculations from FalsePos.py                #
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
#                        Primary Functions and Classes                        #
###############################################################################


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
        self.n_counts     = 0
        self.other_counts = 0

        # Winners
        self.mat_win = 0
        self.pat_win = 0
        self.not_sig = 0
        self.weird   = 0
        self.failed  = 0

        # Significant metrics for all SNPs in this Gene
        self.pval = None
        self.win  = None

    def sum_counts(self):
        """Add up all counts in snps."""
        self.mat_counts   = 0
        self.pat_counts   = 0
        self.n_counts     = 0
        self.other_counts = 0
        for snp in self.snps:
            self.mat_counts   += snp.mat_counts
            self.pat_counts   += snp.pat_counts
            self.n_counts     += snp.n_counts
            self.other_counts += snp.other_counts

    def calc_pval(self):
        """Take the median of the pvalues of all SNPs as our pvalue.

        We only include SNPs with no Ns or non-parental alleles.
        """
        pvals = []
        for snp in self.snps:
            # Don't include confusing SNPs
            #  if len(snp) > 1 and not snp.n_counts and not snp.other_counts:
            if not hasattr(snp, 'pval') and not snp.pval:
                snp.calc_pval()
            pvals.append(snp.pval)
        pvals = [p for p in pvals if isinstance(p, float)]
        if pvals:
            self.pval = numpy.median(pvals)
        else:
            self.pval = numpy.nan
            global GENE_NULL
            GENE_NULL += 1

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
            if snp.win == 'mat':
                self.mat_win += 1
            elif snp.win == 'pat':
                self.pat_win += 1
            elif snp.win == 'NS':
                self.not_sig += 1
            elif snp.win == 'WEIRD':
                self.weird += 1
            else:
                self.failed += 1  # Not bothering with greater res now.

        # Winner must have more than 60% of alleles with gene-level
        # significance
        if not self.pval:
            self.calc_pval()
        if not self.pval:
            logme.log('No pvalue for gene {}'.format(self), 'debug')
            self.win = 'NA'
            return

        if self.pval < THRESH2:
            if self.weird/len(self) > 0.4:
                self.win = 'WEIRD'
            elif self.mat_win > self.pat_win and self.mat_win/len(self) > 0.6:
                self.win = 'mat'
            elif self.pat_win > self.mat_win and self.pat_win/len(self) > 0.6:
                self.win = 'pat'
            else:
                self.win = '?'
        else:
            self.win = 'NS'

    def reset(self):
        """Reset all counts and all SNPs to 0."""
        self.mat_counts   = 0
        self.pat_counts   = 0
        self.n_counts     = 0
        self.other_counts = 0

        # Winners
        self.mat_win = 0
        self.pat_win = 0
        self.not_sig = 0
        self.weird   = 0
        self.failed  = 0

        # Reset snps
        for snp in self.snps:
            snp.reset()

    def __len__(self):
        """How many SNPs are in this gene."""
        return len(self.snps)

    def __repr__(self):
        """Summary info."""
        self.sum_counts()
        return "{}(mat:{};pat:{};N:{};other:{})".format(self.name,
                                                        self.mat_counts,
                                                        self.pat_counts,
                                                        self.n_counts,
                                                        self.other_counts)


###############################
#  Class to hold SNP alleles  #
###############################


class SNP(object):

    """A SNP.

    Contains::
        - name              -- The name or position of this gene
        - mat and pat alleles
        - gene              -- A link to the parent gene
        - counts for::
            - mat_counts
            - pat_counts
            - n_counts      -- Read had an N at this position
            - other_counts  -- Non-matching allele, shouldn't happen
        - pval  -- A binomial pvalue
        - win   -- The winning parent, one of::
            - 'mat'
            - 'pat'
            - 'Ns > #'  -- Too many Ns in read
            - 'WEIRD'   -- Too many non-parental alleles
            - 'NA'      -- Not enough counts
            - 'NS'      -- Non-significant
            - 'homo'    -- Identical counts with significant p-val, shouldn't
                           happen.

    """

    name         = None
    mat_counts   = 0
    pat_counts   = 0
    n_counts     = 0
    other_counts = 0

    pval = None  # A binomial test p-value
    win  = None  # The winning parent.

    def __init__(self, name, mat, pat, gene):
        """Create a SNP object.

        :mat:  The maternal allele
        :pat:  The paternal allele
        :gene: A gene name refering to a Gene object

        """
        allowed_bases = ['A', 'T', 'G', 'C']
        if mat.upper() in allowed_bases:
            self.mat = mat
        else:
            raise Exception('SNP base not allowed')
        if pat.upper() in allowed_bases:
            self.pat = pat
        else:
            raise Exception('SNP base not allowed')
        if not isinstance(gene, Gene):
            raise Exception('exon should be type Exon, is {}'.format(
                type(gene)))
        self.name = name
        self.gene = gene
        self.gene.snps.append(self)
        self.reset()

    def add_count(self, base, count):
        """Add count by comparing base to pat/mat."""
        base = base.upper()
        if base == self.mat:
            self.add_mat(count)
        elif base == self.pat:
            self.add_pat(count)
        elif base == 'N':
            self.add_n(count)
        else:
            self.add_other(count)

    def calc_pval(self):
        """Assign a p-value to the counts based on the binomial test.

        We can't use exactly the same test used in the original pipeline
        because counts within a gene are not independent because many will
        be on the same read and we have paired end data.

        To make up for that we calculate the pvalue using a binomial test
        per SNP and then take the median of all those p-values as the
        p-value for the snps.

        p-value is calculated using a binomial test::
            - the smallest number of counts is the 'number of successes'
            - the sum of both alleles is the 'number of trials'
            - p is the default 'hypothesized probability of success'

        """
        if len(self) > 1 and \
                (float(self.other_counts)/float(len(self)) < 0.1 or
                 float(self.n_counts)/float(len(self)) > NTHRESH):
            smaller = min(self.mat_counts, self.pat_counts)
            both    = self.mat_counts + self.pat_counts

            # Calculate the p-value
            self.pval = binomtest(smaller, both, 0.5)
        else:
            self.pval = numpy.nan
            global SNP_NULL
            SNP_NULL += 1

    def calc_winner(self):
        """Try to pick a winner based on count ratios.

        This is a reimplementation of the AlleleSeq algorithm in testCounts().

        Sets self.win to one of::
            'mat', 'pat', 'homo', 'Ns > #', 'WEIRD', 'NA', 'NS'
        """
        # This is different than AlleleSeq, I treat N counts separately.
        total      = len(self) - self.n_counts
        allelecnts = self.mat_counts + self.pat_counts

        # Reject if total Ns is above NTHRESH or not enough counts
        if len(self) > 2:
            if float(self.n_counts)/float(len(self)) > NTHRESH:
                logme.log(('{}:{} failed thresh1 (counts in alleles/total ' +
                           'counts).\n N counts: {}; total: {}\nSNP:{}').format(
                            self.gene.name, self.name, self.n_counts,
                            len(self), self),
                            'debug')
                self.win = 'Ns > {}'.format(NTHRESH)
                return
        else:
            self.win = 'NA'
            return

        # Find wierd SNPs
        if float(allelecnts)/float(total) < THRESH1:
            logme.log(('{}:{} failed thresh1 (counts in alleles/total ' +
                       'counts).\n Allele counts: {}; total: {}\nSNP:{}')
                       .format(self.gene.name, self.name, allelecnts, total,
                               self),
                      'debug')
            self.win = 'WEIRD'
            return

        # Calculate winners
        if not self.pval:
            self.calc_pval()
        if self.pval is None:  # If there still isn't a pval, skip
            self.win = 'NA'
            return
        # Only calculate winners if p-value is significant
        if self.pval < THRESH2:
            if self.mat_counts > self.pat_counts:
                self.win = 'mat'
            elif self.pat_counts > self.mat_counts:
                self.win = 'pat'
            else:
                self.win = 'homo' # Counts are the same. Shouldn't happen.
        else:
            self.win = 'NS'  # Not significant

    def add_mat(self, count):
        """Add maternal counts."""
        self.mat_counts += count
        self.gene.sum_counts()

    def add_pat(self, count):
        """Add paternal counts."""
        self.pat_counts += count
        self.gene.sum_counts()

    def add_n(self, count):
        """Add counts for N's, to distinguish them from 'other'."""
        self.n_counts += count
        self.gene.sum_counts()

    def add_other(self, count):
        """Add counts that are neither maternal or paternal.

        These generally shouldn't occur, and checking them is valuable.
        """
        self.other_counts += count
        self.gene.sum_counts()

    def reset(self):
        """Reset all counts to 0."""
        self.mat_counts   = 0
        self.pat_counts   = 0
        self.n_counts     = 0
        self.other_counts = 0

    def __len__(self):
        """The total number of SNPs."""
        return self.mat_counts + self.pat_counts \
            + self.n_counts + self.other_counts

    def __repr__(self):
        """Summary info."""
        return "SNP<{}(mat:{};pat:{};N:{};other:{})>".format(
            self.name, self.mat_counts, self.pat_counts, self.n_counts,
            self.other_counts)


#############################
#  House Keeping Functions  #
#############################


def num_to_chrom(chrom):
    """Return a chromsome in the format chr# even if is number."""
    chrom = str(chrom)
    return chrom if chrom.startswith('chr') else 'chr' + chrom


def flip_mat_pat(snps):
    """Take a dictionary of SNPs and flip all mat and pat sites.

    Note: This changes the original dictionary and does not make a copy.
    """
    for chr, poss in snps.items():
        for pos, snp in poss.items():
            mat = snp.mat
            pat = snp.pat
            snp.mat = pat
            snp.pat = mat
    return snps


def create_df(genes):
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
    # Populate dictionaries from every gene
    df_dict = {}
    for name, gene in genes.items():
        df_dict[name] = {'Mat_Counts': gene.mat_counts,
                         'Pat_Counts': gene.pat_counts,
                         'N_Counts':   gene.n_counts,
                         'Others':     gene.other_counts,
                         'SNPs':       len(gene),
                         'Winner':     gene.win,
                         'pval':       gene.pval,
                         'Mat_wins':   gene.mat_win,
                         'Pat_wins':   gene.pat_win,
                         'Not_Sig':    gene.not_sig,
                         'Weird':      gene.weird,
                         'Failed':     gene.failed}
    column_order=['Mat_Counts', 'Pat_Counts', 'N_Counts', 'Others', 'SNPs',
                  'Winner', 'pval', 'Mat_wins', 'Pat_wins', 'Not_Sig',
                  'Weird', 'Failed']
    df = pandas.DataFrame.from_dict(df_dict, orient='index')
    df.index.name = 'GENE'
    df = df[column_order]

    return df


##################
#  File Parsing  #
##################

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


def alleleseq_snp_parser(infile):
    """Iterate through a alleleseq SNP file and yield the line."""
    with open_zipped(infile) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            f = line.split('\t')
            ref, alt = f[5]
            yield (num_to_chrom(f[0]), int(f[1]), '.', ref, alt)


def bed_snp_parser(infile):
    """Iterate through a bed file and yield the line."""
    with open_zipped(infile) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            f = line.split('\t')
            ref, alt = f[4].split('|')
            yield (num_to_chrom(f[0]), int(f[1])+1, f[3], ref, alt)


def vcf_snp_parser(infile):
    """Iterate through a vcf file and yield the line."""
    with open_zipped(infile) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            f = line.split('\t')
            yield (num_to_chrom(f[0]), int(f[1]), f[2], f[3], f[4])


def file_format(infile):
    """Return bed or vcf depending on file type.

    Uses first line of infile to check.
    """
    vcf_headers = ['##fileformat=VCFv4.0', '#CHROM\tPOS\tID']
    with open_zipped(infile) as fin:
        line = fin.readline()
        for head in vcf_headers:
            if line.startswith(head):
                return 'vcf'
        while True:
            if line.startswith('#'):
                line = fin.readline()
                continue
            break
        fields = line.split('\t')
        if len(fields) < 3:
            raise Exception('Not a bed or vcf')
        if fields[2].isdigit():
            if '|' not in fields[4]:
                raise Exception('Looks like a bed file, but fifth column ' +
                                'is not in the format REF|ALT')
            return 'bed'
        if len(fields[3]) == 2:
            return 'as'
        if len(fields[3]) == 1:
            return 'vcf'
        # Identification failed
        raise Exception('File identification failed')


class SNPFile(object):

    """Parse a PHASED input file of SNPs.

    Supports bed or vcf. If bed, the fifth column must be REF|ALT.

    Only works with a SNP file (not genes)

    NOTE: ** Forces 1 based position **

    Yields: (chr, pos, name, ref, alt)
    """

    def __init__(self, infile):
        """Create an iterator from infile."""
        format = file_format(infile)
        if format == 'bed':
            self.file = bed_snp_parser(infile)
        elif format == 'vcf':
            self.file = vcf_snp_parser(infile)
        elif format == 'as':
            self.file = alleleseq_snp_parser(infile)
        else:
            raise Exception('Unrecognized file format')

    def __iter__(self):
        """Iterate through file."""
        return self

    def __next__(self):
        """Iterate through file."""
        return next(self.file)

    # Support python2 also
    next = __next__


def parse_gtf(gtf_file):
    """Return a defaultdict of Chrom objects for lookup.

    To lookup, just run exons[chromsome].find(location) (where exons is the
    defaultdict returned by this function).

    :returns: defaultdict(exons), dict(genes)
    """
    # Initialize the list exons for lookup
    exons = defaultdict(Chrom)

    # Initialize a dictionary of genes
    genes = {}

    # Build regexes
    gene_id  = re.compile(r'gene_id "([^"]+)"')
    trans_id = re.compile(r'transcript_id "([^"]+)"')

    with open(gtf_file) as fin:
        for line in fin:
            fields = line.rstrip().split('\t')
            if not fields[2] == 'exon':
                continue
            chrom  = num_to_chrom(fields[0])
            start  = int(fields[3])
            end    = int(fields[4])
            strand = fields[6]
            gene   = gene_id.findall(fields[8])[0]
            trans  = trans_id.findall(fields[8])[0]
            if gene not in genes:
                genes[gene] = Gene(gene, trans)
            exon = Exon(genes[gene], start, end, strand)
            exons[chrom].append(exon)

    return exons, genes


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
            chrom  = num_to_chrom(fields[0])
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


def parse_snp_file(snp_file, exons):
    """Return a dictionary of SNP objects: {'chr'=>'pos'=>SNP}

    Forces chromsome to be in format chr#.

    :exons: The must be the output of the parse_gtf function.
            It is used to assign an exon to every SNP.

    """
    total    = 0
    in_gene  = 0
    snp_dict = {}
    fin      = SNPFile(snp_file)
    for chrom, loc, name, mat, pat in fin:
        total += 1
        gene  = exons[chrom].find(loc)  # Lookup exon this SNP is in
        if gene is None:
            continue  # Skip all SNPs that aren't in exons.
        in_gene += 1
        # If name isn't defined, name is position.
        if not name or name == '.' or name == 'NA':
            name = chrom + ':' + str(loc)
        # Add the SNP to the dictionary
        if chrom not in snp_dict:
            logme.log('Working on {}'.format(chrom), 'debug')
            snp_dict[chrom] = {}
        snp_dict[chrom][loc] = SNP(name, mat, pat, gene)
    logme.log('{chr}: {count} SNPs of {total} in genes: {per:.2f}%'.format(
        chr=chrom, count=in_gene, total=total, per=(100*in_gene/total)),
              'debug')
    return snp_dict


def split_snpfile(snp_file):
    """Split a SNP file by chromosome, return a list of the new files.

    No longer used, awk is so much faster.
    """
    with open(snp_file) as fin:
        line = fin.readline()
        chrom = line.split('\t')[0]
        outfile = open(chrom + '.snp_file.tmp', 'w')
        outfiles = [chrom + '.snp_file.tmp']
        for line in fin:
            newchr = line.split('\t')[0]
            if newchr == chrom:
                outfile.write(line)
            else:
                outfile.close()
                outfile = open(newchr + '.snp_file.tmp', 'w')
                outfiles.append(newchr + '.snp_file.tmp')
                outfile.write(line)
        outfile.close()
    return outfiles


######################
#  Primary function  #
######################


def get_gene_counts(snps, genes, count_file):
    """Assign gene level counts to all SNPs in count_files.

    :snps:       The snps dictionary from parse_snp_file()
    :genes:      The genes object from parse_gtf()
    :count_file: A single count file from AlleleSeq
    :returns:    A copy of genes with only genes for this individual

    """
    # Initialize snp counters
    total   = 0
    in_gene = 0

    if logme.MIN_LEVEL == 'debug':
        sys.stderr.write('snps keys: {}\n'.format(snps.keys()))

    with open(count_file, 'rb') as fin:
        # This is a dict: {'chr'=>{int('loc')=>{'a':#,'c':#,'g':#,'t':#}}}
        ind_counts = pickle.load(fin)
        for chrom, locs in ind_counts.items():
            for loc, bases in locs.items():
                total += 1
                chrom = num_to_chrom(chrom)
                if chrom not in snps:
                    sys.stderr.write('{} not in SNPs. SNPs chroms: {}\n'.format(
                        chrom, snps.keys()))
                    logme.log('Missing chromosome ' +
                              '{}\nChromosomes: {}'.format(chrom,
                                                           snps.keys()),
                              'critical')
                    continue
                if loc not in snps[chrom]:
                    continue
                in_gene += 1
                snp = snps[chrom][loc]

                #####################################
                #  Actually assign the counts here  #
                #####################################

                for base, count in bases.items():
                    base = base.upper()  # All comparisons uppercase
                    snp.add_count(base, count)
                    # As we are looking for significance per SNP per individual
                    # we can calculate the binomial p-value right now.
                    # NOTE: skipping this for now, as doing it at gene level.
                    snp.calc_pval()
                    snp.calc_winner()

    genes = {}
    for chr, poss in snps.items():
        for pos, snp in poss.items():
            genes[snp.gene.name] = snp.gene
    # Return a copy of the genes object
    logme.log('Copying genes dictionary', 'debug')
    ind_genes = deepcopy(genes)
    logme.log('Summing counts', 'info')
    for gene in ind_genes.values():
        gene.sum_counts()
    logme.log('Calculating pvalues', 'info')
    for gene in ind_genes.values():
        gene.calc_pval()
    logme.log('Calculating winners', 'info')
    for gene in ind_genes.values():
        gene.calc_winner()

    # Write out stats
    sys.stderr.write('In file {}, {} SNPs of {} '.format(count_file,
                                                         in_gene, total) +
                     'in genes. {:.2f}%\n'.format(100*in_gene/total))

    # Reset counts
    if logme.MIN_LEVEL == 'debug':
        mat_counts = 0
        pat_counts = 0
        n_counts = 0
        other_counts = 0
        for name, gene in ind_genes.items():
            mat_counts += gene.mat_counts
            pat_counts += gene.pat_counts
            n_counts += gene.n_counts
            other_counts += gene.other_counts
        logme.log('mat: {}; pat: {}; n: {}; other: {}'.format(
            mat_counts, pat_counts, n_counts, other_counts), 'debug')
    for gene in genes.values():
        gene.reset()
    return ind_genes


def snps_to_genes(snps):
    """Convert a snps dict to a genes dict."""
    genes = {}
    for chrom in snps.values():
        for snp in chrom.values():
            if snp.gene.name not in genes:
                genes[snp.gene.name] = snp.gene
    return genes


###############################################################################
#                               Run as a script                               #
###############################################################################


def main(argv=None):
    """Command line parsing."""

    usage  = ' alleleseq_genes create|analyze -h'
    cusage = ' alleleseq_genes create  [-j] [-l] [-v|-q] -o snp_pickle gtf_file snp_file'
    ausage = ' alleleseq_genes analyze [-j] [-l] [-v|-q] [-f fdr_cutoff] [-c columns] [-o out.tsv] snp_pickle [1.cnt [2.cnt ...]]'
    try:
        usage += '\n       {}\n       {}\n'.format(cusage, ausage)
    except ValueError:
        sys.stderr.write('Python 2.7+ required\n')
        return 1

    create_mode = dedent("""\
        Inputs
        ------

        gtf_file:   A simple GTF or BED format file with all genes of interest.
        snp_file:   snps.txt file created by AlleleSeq in the format::
                        1\\t3000715\\tT\\tTC\\tGC\\tTG\\tPHASED
                    Only column [5] (0-based) is used to determine mat/pat for
                    that position. In this case the result is::
                        mat: T, pat: G

        Output
        ------
        snp_pickle: Created by the analyze mode, a hybrid object containing
                    all SNPs with maternal and paternal genotypes and gene
                    level information. Used for counting gene counts.

        Output defaults to the snp_file with a '.pickle' suffix.""")

    analyze_mode = dedent("""\
        Inputs
        ------
        snp_pickle:  Created by the analyze mode, a hybrid object containing
                     all SNPs with maternal and paternal genotypes and gene
                     level information. Used for counting gene counts.
        cnt files:   .cnt files created by AlleleSeq, these are pickle objects.
        counts file: The combined counts file created by CombineSNPCounts.py

        Outputs
        -------

        In individual mode, the first column will be the individual name, taken
        from the name of the cnt file.

        In both modes, the remaining columns will be::
            GENE: The name of the gene
            TX: The name of the transcript
            MAT_COUNTS: The sum of all maternal SNPs across the gene
            PAT_COUNTS: The sum of all paternal SNPs across the gene
            WIN: M|P (maternal or paternal)
            P: The binomial pvalue.
            ... : Any extra columns from the columns file will be added here

        If --pandas is specified, a pickled pandas dataframe is output.

        Output defaults to STDOUT.

        If an fdr_cutoff is provided, the output is filtered to only inlude
        genes that beat that cutoff.""")

    if not argv:
        argv = sys.argv[1:]

    mode_desc = '\nMode:\n  create|analyze'
    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Allow subparsers
    subparsers = parser.add_subparsers(title='Mode', dest='mode')
    common     = argparse.ArgumentParser(usage=None, add_help=False)
    multip     = argparse.ArgumentParser(usage=None, add_help=False)
    subparsers.required = True

    ## Common Options ##

    # Multiprocessing
    multi = multip.add_argument_group('Multi-threading')
    multi.add_argument('-j', '--jobs', type=int, metavar='',
                       help="Run this many threads, default is " +
                       "single-threaded. To use all cores on this machine, "
                       "provide -1.")

    # Logging
    logging = common.add_argument_group('Logging')
    logging.add_argument('-l', '--logfile', default=sys.stderr,
                         help="Default STDERR")
    logging.add_argument('-v', '--verbose', action='store_true',
                         help="Write more messages")
    logging.add_argument('-q', '--quiet', action='store_true',
                         help="Only print warnings")

    # Help
    helparg = common.add_argument_group('Help')
    helparg.add_argument('-h', '--help', action='help',
                         help="Show this help message and exit.")

    ## Mode based options ##

    # Create SNP object
    create  = subparsers.add_parser(
        'create', description='Create the SNPs object',
        epilog=create_mode, parents=[multip, common], usage=cusage,
        add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    # File handling
    cfiles = create.add_argument_group('Input and Output Files')
    cfiles.add_argument('gtf_file',
                        help="A GFF/GTF or bed file with gene locations.")
    cfiles.add_argument('snp_file',
                        help="An AlleleSeq SNP File.")
    cfiles.add_argument('-o', dest='outfile', metavar='snps.pickle',
                        help="The output file, default is <snp_file>.pickle.")

    # Analyze Counts
    analyze  = subparsers.add_parser(
        'analyze', description='Analyze the counts',
        epilog=analyze_mode, parents=[multip, common], usage=ausage,
        add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Mode
    #  mode = analyze.add_argument_group('Analyze Mode')
    #  mode.add_argument('-m', dest='mode', choices=['ind', 'comb'],
                      #  default='ind', help='ind: run on individual .cnt files. ' +
                      #  'comb: run on a single counts.txt file')

    # File handling
    files = analyze.add_argument_group('Input and Output Files')
    files.add_argument('snp_pickle',
                       help="The SNPs file output by analyze mode.")
    files.add_argument('count_files', nargs='+',
                       help="The count files from AlleleSeq.")
    files.add_argument('-c', '--columns', metavar='cols.txt',
                       help="A tab delimited file with extra columns to " +
                       "add to output, first column must be same as in " +
                       "gtf_file.")
    files.add_argument('-o', dest='outfile', metavar='out.tsv',
                       help="The output file, default STDOUT.")
    files.add_argument('--data',
                       help="Output a pickled dictionary of raw data with useful methods to this file.")
    files.add_argument('--pandas',
                       help="Output a pickled pandas dataframe to this file.")

    # FDR Calulations
    fdrcalc = analyze.add_argument_group('FDR Calculation')
    fdrcalc.add_argument('--filter-fdr', action='store_true',
                         help="Filter the output by FDR")
    fdrcalc.add_argument('-f', '--fdr-cutoff', type=float, default=0.1,
                         metavar='', help="FDR cutoff (Default 0.1).")
    fdrcalc.add_argument('-s', '--simulations', type=int, default=10,
                         metavar='',
                         help="# simulations for FDR calculation " +
                         "(Default: 10)")

    # Flipping a SNP dictionary

    flipusage = ' alleleseq_genes flip infile outfile'

    flipdesc = dedent("""\
    Flip all SNP counts in a SNP dictionary so that the
    paternal allele is now the maternal allele and visa-versa.
    Takes the output of create mode and outputs a flipped dictionary
    in the same format.
    """)

    flip  = subparsers.add_parser(
        'flip', description=flipdesc, parents=[common], add_help=False,
        usage=flipusage, formatter_class=argparse.RawDescriptionHelpFormatter)

    ffiles = flip.add_argument_group('Input and Output Files')
    ffiles.add_argument('input',
                        help="The SNP pickle output of create mode.")
    ffiles.add_argument('output',
                        help="The name of the output file.")

    args = parser.parse_args(argv)

    # Logging
    logme.LOGFILE = args.logfile
    if args.verbose:
        logme.MIN_LEVEL = 'debug'
    elif args.quiet:
        logme.MIN_LEVEL = 'warn'

    # Check arguments
    if args.mode == 'create':
        with open(args.snp_file) as fin:
            line = fin.readline().rstrip().split('\t')
            assert len(line) == 7
            assert line[6] == 'PHASED' or line[6] == 'HOMO'
            assert len(line[5]) == 2

        gtf_name = args.gtf_file.split('.')
        if 'gtf' in gtf_name or 'gff' in gtf_name:
            with open(args.gtf_file) as fin:
                line = fin.readline().rstrip().split('\t')
                assert len(line) == 9
                assert 'gene_id' in line[8]
        elif 'bed' in gtf_name:
            with open(args.gtf_file) as fin:
                line = fin.readline().rstrip().split('\t')
                assert len(line) >= 6
                assert line[2].isdigit()
        else:
            sys.stderr.write('Gene file must be bed/gtf/gff.\n')
            return 6

    elif args.mode == 'analyze':
        for cnt_file in args.count_files:
            if not os.path.isfile(cnt_file):
                sys.stderr.write('No such file: {}\n'.format(cnt_file))
                return 9
            with open(cnt_file) as fin:
                if not fin.readline().rstrip() == '(dp1':
                    sys.stderr.write('Count file {} does '.format(cnt_file) +
                                     'not look like a count file\n')
                    return 2

    elif args.mode != 'flip':
        print(args.mode)
        sys.stderr.write('usage: {}'.format(usage))
        sys.stderr.write('Mode (create|analyze|flip) required.\n')
        return 1

    # Set up multiprocessing
    if args.mode != 'flip' and args.jobs:
        jobs = {}
        logme.log('Multiplexing with {} threads.'.format(args.jobs), 'debug')
        pool = Pool(args.jobs)

    #################
    #  Create Mode  #
    #################

    if args.mode == 'create':
        snp_pickle = args.outfile if args.outfile else \
            '.'.join(args.snp_file.split('.')[:-1]) + '.pickle'

        ### Get exon info from GTF File###
        # This function loops through the gtf file and creates an Exon() record
        # for every exon and a Gene() record for every gene. The Exon records
        # all contain references to their parent gene, they are returned as a
        # defaultdict of Chrom() objects with a find method so that a SNP can
        # be placed in an exon and thus gene by just running::
        #   gene = exons[chr].find(location, strand)
        # The gene object has the two methods: add_mat() and add_pat(), which
        # will increment the count of mat or pat variables. All the genes are
        # indexed by name and are held in the genes dictionary.
        logme.log('Parsing gene file {}'.format(args.gtf_file), 'info',
                  also_write='stderr')
        if 'gtf' in gtf_name or 'gff' in gtf_name:
            exons, genes = parse_gtf(args.gtf_file)
        elif 'bed' in gtf_name:
            exons, genes = parse_gene_bed(args.gtf_file)
        logme.log('{} genes total'.format(len(genes)), 'info')
        if args.verbose:
            sys.stderr.write('{} total chromosomes with exons:\n'.format(
                len(exons)))
            max_len = 8
            for chrom in exons.keys():
                max_len = max(len(chrom), max_len)
            max_len += 1
            for chrom, poss in sorted(exons.items()):
                sys.stderr.write('\t{}{}\n'.format(chrom.ljust(max_len),
                                                   len(poss)))

        # Get SNP info -- just returns a dictionary of SNPs to hold maternal and
        # paternal alleles. This is slow, so parallelize and save with pickle.
        # Even more time can be saved by pre-filtering the bed file with
        # github.com/TheFraserLab/ASEr/blob/master/bin/filter_snps_by_exon
        logme.log('Parsing SNPs {}'.format(args.snp_file), 'info')
        if args.jobs:
            logme.log('Splitting SNP file', 'debug')
            #  snp_files = split_snpfile(args.snp_file)
            # Delete old files first
            for snp_file in [i for i in os.listdir('.') \
                             if i.endswith('.snp_file.tmp')]:
                os.remove(snp_file)
            # We need to sleep a moment or the script deletes chromosome 3,
            # no idea why.
            sleep(2)
            check_call('awk \'{print $0 >> $1".snp_file.tmp"}\' ' +
                       args.snp_file, shell=True)
            snp_files = [i for i in os.listdir('.') if i.endswith('.snp_file.tmp')]
            logme.log('Splitting done', 'debug')
            logme.log('Submitting {} jobs.'.format(len(snp_files)))
            for snp_file in snp_files:
                snps = {}
                jobs[snp_file] = pool.apply_async(parse_snp_file, (snp_file,
                                                                   exons))
            for tmpfile, job in jobs.items():
                snps.update(job.get())
                try:
                    os.remove(tmpfile)
                except OSError:
                    pass
        else:
            snps = parse_snp_file(args.snp_file, exons)

        # Save the output so we don't have to do this again.
        logme.log('Creating pickle of SNPs')
        with open(snp_pickle, 'wb') as fout:
            pickle.dump(snps, fout)

        logme.log('SNP parsing complete', 'info')

    ##################
    #  Analyze Mode  #
    ##################

    elif args.mode == 'analyze':
        logme.log('Reading SNP file', 'info', also_write='stderr')

        # snps is {'chr'=>'position'=>SNP}
        with open(args.snp_pickle, 'rb') as fin:
            snps = pickle.load(fin)

        logme.log('Building gene list.', 'info', also_write='stderr')
        genes = {}
        for chrom in snps.values():
            for snp in chrom.values():
                if snp.gene.name not in genes:
                    genes[snp.gene.name] = snp.gene

        logme.log('{} chromosomes with SNPs'.format(len(snps)), 'debug')
        if logme.MIN_LEVEL:
            total = 0
            max_len = 0
            for chrom in snps.keys():
                max_len = max(max_len, len(chrom))
            max_len += 2
            for chrom, poss in sorted(snps.items()):
                sys.stderr.write('\t{chrom:{len}}{pos}\n'.format(
                    chrom=chrom, len=max_len, pos=len(poss)))
                total += len(poss.values())
        logme.log('{} total SNPs\n'.format(total), 'debug')

        # Loop through the count files and add counts to genes
        individuals = {}
        for count_file in args.count_files:
            ind_name = count_file.split('.')[0]  # Take name from file name
            logme.log('Building gene list for {}'.format(ind_name), 'debug')
            funcargs = (snps, genes, count_file)
            if args.jobs and len(args.count_files) > 1:
                jobs[ind_name] = pool.apply_async(get_gene_counts, funcargs)
            else:
                individuals[ind_name] = get_gene_counts(*funcargs)
        if args.jobs and len(args.count_files) > 1:
            for ind_name, job in jobs.items():
                individuals[ind_name] = job.get()

        if args.data:
            logme.log('Writing unflitered data to {}'.format(args.data),
                      'info')
            with open(args.data, 'wb') as fout:
                pickle.dump(individuals, fout)

        ############################
        #  Make pandas dataframes  #
        ############################

        inds = {}
        logme.log('Making pandas dataframes', 'info')
        for ind, genes in individuals.items():
            inds[ind] = create_df(genes)

        ####################################
        #  Calculate FDR and filter genes  #
        ####################################

        logme.log('Calculating p-values at FDR {}'.format(args.fdr_cutoff),
                  'info')
        fdr_pvals = {}
        for ind, df in inds.items():
            fdr_pvals[ind] = calc_fdr(
                [tuple(x) for x in df[['Mat_Counts', 'Pat_Counts', 'pval']].values],
                target=args.fdr_cutoff, sims=args.simulations,
                verbose=args.verbose)
            logme.log('In ind {} p-values smaller than {} beat FDR of {}'
                      .format(ind, fdr_pvals[ind], args.fdr_cutoff), 'info')


        # Filter by FDR if requested
        if args.filter_fdr:
            logme.log('Filtering genes by FDR less than {}'.format(
                      args.fdr_cutoff), 'info')
            filtered = {}
            for ind, df in inds.items():
                logme.log('Filtering {}'.format(ind), 'debug')
                filtered[ind] = df[df.pval < fdr_pvals[ind]]
            inds = filtered

        # Create merged df
        newdf = {}
        for ind, df in inds.items():
            cols = ['Ind'] + list(df.columns)
            df['Ind'] = pandas.Series([ind for i in range(0, len(df))],
                                      index=df.index)
            newdf[ind] = df[cols]
        logme.log('Merging dictionaries', 'info')
        master_df = pandas.concat([d for d in newdf.values()])

        # Save pandas dfs
        if args.pandas:
            logme.log('Writing pandas dataframe to {}'.format(args.pandas),
                      'info')
            with open(args.pandas, 'wb') as fout:
                pickle.dump(master_df, fout)

        # Write the output
        logme.log('Writing output to {}'.format(args.outfile), 'info')
        master_df.to_csv(args.outfile, sep='\t', float_format='%.3e')


    ###############
    #  Flip Mode  #
    ###############
    elif args.mode == 'flip':
        logme.log('Loading SNPs from {}'.format(args.input), 'debug')
        with open(args.input, 'rb') as fin:
            snps = pickle.load(fin)
        logme.log('Flipping mat and pat', 'debug')
        snps = flip_mat_pat(snps)
        logme.log('Writing new snps to {}'.format(args.output), 'debug')
        with open(args.output, 'wb') as fout:
            pickle.dump(snps, fout)

    logme.log('DONE', 'info', also_write='stderr')

# The End.
if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
