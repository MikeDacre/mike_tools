#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert a bed file with quality info into an AlleleSeq COV file.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2014-07-21 17:06
 Last modified: 2016-03-25 12:16

============================================================================
"""
from sys import stdin, stdout

stdout.write("chrm\tsnppos\trd\n")
for i in stdin:
    f = i.rstrip().split('\t')
    stdout.write('\t'.join([f[0], f[1], f[4]]) + '\n')
