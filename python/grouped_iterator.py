#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Iterate through a pre-sorted text file and return lines as a group.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-29-27 16:09
 Last modified: 2016-09-27 17:07

============================================================================
"""
import os
import sys
import gzip
import bz2


def giterate(infile, groupby, columns=None, sep='\t', header=False,
             pandas=False):
    """Iterate through a text file and yield lines in groups.

    :infile:  The path to a plain text, gzipped, or bzipped text file or a file
              handle.
    :groupby: An integer reference to the column you wish to group on.
    :columns: Either None, or an integer count of columns, or a list of column
              names you would like to use to access your data. If integer is
              provided then column count is confirmed.
    :header:  If true, first line is used as column names if none provided or
              skipped.
    :pandas:  Yield a pandas dataframe for every group instead of a list of
              lists or Line objects.
    :yields:  Default is a list of lists for each group. If pandas is True,
              then yields a dataframe for every group.

    """
    if pandas:
        import pandas as pd

    if isinstance(columns, list):
        collen  = len(columns)
        useline = True
    else:
        collen  = columns if isinstance(columns, int) else None
        useline = False
        columns = None

    with open_zipped(infile) as fin:
        grp = []
        nxt = ''
        if header:
            head = fin.readline()
            if not columns:
                columns = head.rstrip().split(sep)
        for line in fin:
            fields = fin.rstrip().split(sep)
            if collen:
                assert collen == fields
            if not nxt:
                nxt = fields[groupby]
                grp.append(fields)
                continue
            if fields[groupby] == nxt:
                grp.append(fields)
                continue
            else:
                if pandas:
                    out = pd.DataFrame(grp)
                    if columns:
                        df.columns = columns
                else:
                    out = grp
                grp = [fields]
                yield out


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
