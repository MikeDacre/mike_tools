#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Investigate linkage disequilibrium between pairs of SNPs.

Uses the LDLink API to search for pairs of SNPs and builds a simple class
(SNP_Pair) with the results.

Info
----
Author: Michael D Dacre, mike.dacre@gmail.com
Organization: Stanford University
License: MIT License, property of Stanford, use as you wish
Created: 2017-21-21 10:04
Version: 0.1a

LDLink
------
https://analysistools.nci.nih.gov/LDlink
https://github.com/CBIIT/nci-webtools-dceg-linkage

This tool uses the above API entirely, thanks to Mitchell Machiela and the
Chanock lab for writing that tool!

Citation
~~~~~~~~
Machiela MJ, Chanock SJ. LDlink a web-based application for exploring
population-specific haplotype structure and linking correlated alleles of
possible functional variants. Bioinformatics. 2015 Jul 2. PMID: 26139635.

Examples
--------
>>>snp = compare_two_variants('rs11121194', 'rs1884352', ['ESN'])

>>>snp
SNP_Pair<rs11121194(['C', 'T']:rs1884352(['G', 'A']) R2: 0.9222 (P: <0.0001)>

>>>snp.rsquared
0.922

>>>snp.lookup_other(1, 'C')
'G'

>>>snp.lookup_other('rs11121194', 'C')
'G'
===============================================================================
"""
__version__ = '0.1a'

import re as _re
from time import sleep as _sleep
from urllib.request import urlopen as _get

import pandas as _pd
from numpy import nan as _nan

SLEEP_TIME = 1.0


###############################################################################
#                          Class to hold the linkage                          #
###############################################################################


class SNP_Pair(object):

    """Association information from a pair of SNPs, created from LDpair.

    Attributes
    ----------
    snp1 : str
    snp2 : str
    chrom :  str
    loc1 : int
        Position of snp1 on chromosome
    loc2 : int
        Position of snp1 on chromosome
    dprime : float_or_nan
    rsquared : float_or_nan
    chisq : float_or_nan
    p : float_or_nan
    p_str : str
        String representation of the p-value
    populations : list
        List of 1000genomes populations
    table : pandas.DataFrame
        Pandas DataFrame of allele counts for both SNPs in the above
        populations. Rows are snp1, columns are snp2. Multiindexed.
    alleles : dict
        Dictionary of alleles by snp
    lookup : dict
        Dictionary of allele in other SNP given an allele in one SNP.

    Methods
    -------
    lookup_other(self, snp, allele)
        Get the allele of a SNP given an allele in the other SNP.

    """

    def __init__(self, x):
        """Parse an input string from LDpair."""
        self.input_string = x
        f = x.split('\n')
        assert f[0] == 'Query SNPs:'
        assert f[3] == ''

        self.snp1, self.chrom, self.loc1 = get_snps(f[1])
        self.snp2, _, self.loc2 = get_snps(f[2])
        self.populations = f[4].split(' ')[0].split('+')

        snps = _re.split(r' +', f[6].strip())
        s1, i1, i2, _, _ = _re.split(r' +[|(] *', f[8].strip(' )'))
        s2, i3, i4, _, _ = _re.split(r' +[|(] *', f[10].strip(' )'))

        cols = _pd.MultiIndex.from_tuples([(self.snp2, i) for i in snps])
        indx = _pd.MultiIndex.from_tuples([(self.snp1, i) for i in [s1, s2]])
        self.table = _pd.DataFrame(
            [[int(i1), int(i2)],
             [int(i3), int(i4)]],
            columns=cols, index=indx
        )

        self.dprime = f[20].strip().split(':')[1].strip()
        try:
            self.dprime = float(self.dprime)
        except ValueError:
            self.dprime = _nan
        self.rsquared = f[21].strip().split(':')[1].strip()
        try:
            self.rsquared = float(self.dprime)
        except ValueError:
            self.rsquared = _nan
        self.chisq = f[22].strip().split(':')[1].strip()
        try:
            self.chisq = float(self.dprime)
        except ValueError:
            self.chisq = _nan
        p = f[23].strip().split(':')[1].strip()
        try:
            self.p = 0.0 if p == '<0.0001' else float(p)
        except ValueError:
            self.p = _nan
        self.p_str = p

        s1a, a1a, s2a, a2a = correlation_lookup.findall(f[25])[0]
        s1b, a1b, s2b, a2b = correlation_lookup.findall(f[26])[0]
        a1a = a1a.upper()
        a1b = a1b.upper()
        a2a = a2a.upper()
        a2b = a2b.upper()
        assert s1a == self.snp1
        assert s1b == self.snp1
        assert s2a == self.snp2
        assert s2b == self.snp2
        self.snp1_alleles = [a1a, a1b]
        self.snp2_alleles = [a2a, a2b]
        self.alleles = {self.snp1: self.snp1_alleles, self.snp2: self.snp2_alleles}

        self.lookup = {
            self.snp1: {
                a1a: a2a,
                a1b: a2b,
            },
            self.snp2: {
                a2a: a1a,
                a2b: a1b,
            },
        }

    def lookup_other(self, snp, allele):
        """Return the linked allele for a given snp.

        Parameters
        ----------
        snp : int_or_str
            Either 1, 2 for SNP 1/2 or rsID.
        allele :str
            The allele for snp.

        Returns
        -------
        str
            Linked allele for other SNP.
        """
        if isinstance(snp, int):
            if snp == 1:
                snp = self.snp1
            elif snp == 2:
                snp = self.snp2
            else:
                raise ValueError('Invalid value for SNP')
        if snp not in [self.snp1, self.snp2]:
            raise ValueError('SNP must be one of {}'.format([self.snp1, self.snp2]))

        allele = allele.upper()

        if allele not in self.alleles[snp]:
            raise ValueError('Allele {} is invalid for SNP {}, possible values are {}'
                             .format(allele, snp, self.alleles[snp]))

        return self.lookup[snp][allele]

    def __repr__(self):
        """Return infomation"""
        return "SNP_Pair<{snp1}({snp1_a}:{snp2}({snp2_a}) R2: {r2} (P: {P})>".format(
            snp1=self.snp1, snp1_a=self.snp1_alleles,
            snp2=self.snp2, snp2_a=self.snp2_alleles,
            r2=self.rsquared, P=self.p_str
        )

    def __str__(self):
        """Print summary"""
        return (
            'Chromosome: {}\n'.format(self.chrom) +
            "SNP1: {} ({}), Alleles: {}\n".format(
                self.snp1, self.loc1, self.snp1_alleles) +
            "SNP2: {} ({}), Alleles: {}\n".format(
                self.snp2, self.loc2, self.snp2_alleles) +
            "Population(s): {}\n".format(self.populations) +
            "Associations (rows are SNP1, columns are SNP2):\n" +
            "\n{}\n\n".format(self.table) +
            "R\u00b2: {}\n".format(self.rsquared) +
            "D': {}\n".format(self.dprime) +
            "Chi-Squared: {}\n".format(self.chisq) +
            "P-value: {}\n".format(self.p_str)
        )


###############################################################################
#                                Main Functions                               #
###############################################################################


def compare_variants(snp_list: list, populations: list, rsquared: float=None):
    """Yield SNP_Pair objects for every pair of variants in snp_list.

    Will wait some amount of time between calls, defined by SLEEP_TIME.

    Parameters
    ----------
    snp_list : list of tuples
        Format: [(var1, var2), ...]

    populations : list
        List of 1000genomes populations (e.g. YRI, ESN), string is converted to
        a list (allowing single population lookup).

    rsquared : float
        An R-squared cutoff to use for returning results, those with an
        r-squared below that value will not be returned. Note that as the
        query must be executed prior to checking the r-squared, a large number
        of requests with low r-squared values will result in long pauses prior
        to a value being yielded.

    Yields
    ------
    SNP_Pair
        list of SNP_Pair objects for each pair of variants in snp_list.
    """
    assert isinstance(snp_list, list)
    assert isinstance(snp_list[0], tuple)
    assert len(snp_list[0]) == 2
    for snp1, snp2 in snp_list:
        snp_pair = compare_two_variants(snp1, snp2, populations)
        if rsquared and isinstance(rsquared, float):
            if snp_pair.rsquared < rsquared:
                print('{} failed r-squared cutoff'.format(repr(snp_pair)))
                _sleep(SLEEP_TIME)
                continue
        yield snp_pair
        _sleep(SLEEP_TIME)


def compare_two_variants(var1: str, var2: str, populations: list) -> SNP_Pair:
    """Return a SNP_pair class for any two rsids.

    Uses the LDpair API:
        https://analysistools.nci.nih.gov/LDlink/?tab=ldpair

    Parameters
    ----------
    var1/var2 : str
        rsIDs to compare
    populations : list
        list of 1000genomes populations (e.g. YRI, ESN), string is converted to
        a list (allowing single population lookup).

    Returns
    -------
    SNP_Pair
        A SNP_Pair class with information about SNP linkage.
    """
    assert isinstance(var1, str)
    assert isinstance(var2, str)
    if isinstance(populations, str):
        populations = [populations]

    req = _get(
        'https://analysistools.nci.nih.gov/LDlink/LDlinkRest/' +
        'ldpair?var1={}&var2={}&pop={}'.format(
            var1, var2, '%2B'.join(populations)
        )
    )

    return SNP_Pair(req.read().decode())


###############################################################################
#                                   Helpers                                   #
###############################################################################


# Parse allele linkage data from last two lines.
correlation_lookup = _re.compile(
    r'^(rs[0-9]+)\(([ATGC])\) allele is correlated with ' +
    r'(rs[0-9]+)\(([ATGC])\) allele$'
)


def get_snps(x: str) -> tuple:
    """Parse a SNP line and return name, chromsome, position."""
    snp, loc = x.split(' ')
    chrom, position = loc.strip('()').split(':')
    return snp, chrom, int(position)
