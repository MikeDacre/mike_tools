#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Take a list of genome positions and return the dinucleotides around it.

For each position, will generate a list of + strand dinucleotides and - strand
dinucleotides.

       Created: 2017-07-27 12:02
 Last modified: 2017-03-14 16:23
"""
import os
import bz2
import gzip
from datetime import timedelta as _td

import fyrd
from Bio import SeqIO as seqio

hg19 = "/godot/genomes/human/hg19"


###############################################################################
#                               Core Algorithm                                #
###############################################################################


def get_dinucleotides(positions, genome_file, base=0, as_dict=False):
    """Return a list of all + and - strand dinucleotides around each position.

    Will loop through each chromosome and search all positions in that
    chromosome in one batch. Lookup is serial per chromosome.

    Args:
        positions (dict):  Dictionary of {chrom->positons}
        genome_file (str): Location of a genome fasta file or directory of
                           files. If directory, file names must be
                           <chrom_name>.fa[.gz]. Gzipped OK.
        base (int):        Either 0 or 1, base of positions in your list
        as_dict (bool):    Return a dictionary of:
                           {chrom->{postion->{'ref': str, '+': tuple, '-': tuple}}}
                           Otherwise just returns two lists with no positions.

    Returns:
        (list, list): + strand dinucleotides, - strand dinucleotides. Returns
                      a dict instead if as_dict is True, see above for info.
    """
    if os.path.isdir(genome_file):
        chroms = positions.keys()
        files = []
        for chrom in chroms:
            files.append(get_fasta_file(genome_file, chrom))
        final = {} if as_dict else ([], [])
        for chrom, fl in zip(chroms, files):
            if as_dict:
                pos = {chrom: positions[chrom]}
                final.update(get_dinucleotides(pos, fl, base, as_dict))
            else:
                plus, minus = get_dinucleotides(pos, fl, base, as_dict)
                final[0] += plus
                final[1] += minus
        return final

    done = []
    results = {} if as_dict else ([], [])
    with open_zipped(genome_file) as fasta_file:
        for chrom in seqio.parse(fasta_file, 'fasta'):
            if chrom.id not in positions:
                continue
            else:
                done.append(chrom.id)
                if as_dict:
                    results[chrom.id] = {}
            for pos in positions[chrom.id]:
                pos    = pos-base
                ref    = chrom[pos]
                plus1  = chrom[pos-1:pos+1]
                plus2  = chrom[pos:pos+2]
                minus1 = plus1.reverse_complement()
                minus2 = plus2.reverse_complement()
                if as_dict:
                    results[chrom.id][pos] = {
                        'ref': ref,
                        '+': (seq(plus1), seq(plus2)),
                        '-': (seq(minus1), seq(minus2))}
                else:
                    results[0] += [plus1, plus2]
                    results[1] += [minus1, minus2]
    if len(done) != len(positions.keys()):
        print('The following chromosomes were not in files: {}'
              .format([i for i in positions if i not in done]))

    return results


###############################################################################
#                               Parallelization                               #
###############################################################################


def get_dinucleotides_parallel(positions, genome_file, base=0, as_dict=False):
    """Return a list of all + and - strand dinucleotides around each position.

    Will loop through each chromosome and search all positions in that
    chromosome in one batch. Lookup is parallel per chromosome.

    Args:
        positions (dict):  Dictionary of {chrom->positons}
        genome_file (str): Location of a genome fasta file or directory of
                           files. If directory, file names must be
                           <chrom_name>.fa[.gz]. Gzipped OK. Directory is
                           preferred in parallel mode.
        base (int):        Either 0 or 1, base of positions in your list
        as_dict (bool):    Return a dictionary of:
                               {chrom->{postion->{'+': tuple, '-': tuple}}}
                           Otherwise just returns two lists with no positions.

    Returns:
        (list, list): + strand dinucleotides, - strand dinucleotides. Returns
                      a dict instead if as_dict is True, see above for info.
    """
    outs = []
    for chrom in positions.keys():
        if os.path.isdir(genome_file):
            fa_file = get_fasta_file(genome_file, chrom)
        if not os.path.isfile(fa_file):
            raise FileNotFoundError('{} not found.'.format(genome_file))
        mins = int(len(positions[chrom])/2000)+45
        time = str(_td(minutes=mins))
        outs.append(
            fyrd.submit(
                get_dinucleotides,
                ({chrom: positions[chrom]}, fa_file, base, as_dict),
                cores=1, mem='6GB', time=time,
            )
        )

    final = {} if as_dict else ([], [])

    for out in outs:
        if as_dict:
            res = out.get()
            final.update(res)
        else:
            plus, minus = out.get()
            final[0] += plus
            final[1] += minus

    return final


###############################################################################
#                              Helper Functions                               #
###############################################################################


def seq(sequence):
    """Convert Bio.Seq object to string."""
    return str(sequence.seq.upper())


def get_fasta_file(directory, name):
    """Look in directory for name.fa or name.fa.gz and return path."""
    fa_file = os.path.join(directory, name + '.fa')
    gz_file = fa_file + '.gz'
    if os.path.isfile(fa_file):
        genome_file = fa_file
    elif os.path.isfile(gz_file):
        genome_file = fa_file
    else:
        raise FileNotFoundError(
            'No {f}.fa or {f}.fa.gz file found in {d}'.format(
                f=name, d=directory
            )
        )
    return genome_file


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


###############################################################################
#                                Run On Files                                 #
###############################################################################


def parse_location_file(infile, base=None):
    """Get a compatible dictionary from an input file.

    Args:
        infile (str): Path to a bed, vcf, or tsv. If tsv should be chrom\\tpos.
                      Filetype detected by extension. Gzipped/B2zipped OK.
        base (int):   Force base of file, if not set, bed/tsv assumed base 0,
                      vcf assumed base-1

    Returns:
        dict: A dict of {chrom->pos}
    """
    if not isinstance(base, int):
        base = 1 if 'vcf' in infile.split('.') else 0
    out = {}
    for chrom, pos in tsv_bed_vcf(infile, base):
        if chrom not in out:
            out[chrom] = []
        out[chrom].append(pos)
    return out


def tsv_bed_vcf(infile, base=0):
    """Interator for generic tsv, yields column1, column2 for every line.

    column1 is assumed to be string, column2 is converted to int and base is
    subtracted from it.
    """
    with open_zipped(infile) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            f = line.rstrip().split('\t')
            yield f[0], int(f[1])-base
