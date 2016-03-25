#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combine AlleleSeq counts.txt and FDR.txt files into a TSV.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-03-25 12:03
 Last modified: 2016-03-25 12:07

   DESCRIPTION: Will run on a list of counts files and a list of fdr files,
                which **MUST BE IN THE SAME ORDER**. If no fdr files are
                provided, the script will attempt to find them by looking
                for a file with the same name as the count file, but ending
                with fdr.txt or FDR.txt instead of .txt.

                If a GTF file of gene locations is provided, a GENE column
                is added to the output.

                Output is handled by pandas.

============================================================================
"""
import os
import sys

args
