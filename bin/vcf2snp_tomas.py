#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert VCF format into AlleleSeq SNPs text format.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2014-07-21 10:40
 Last modified: 2016-03-25 12:13

============================================================================
"""
from sys import stdin, stdout

for i in stdin:
    if i.startswith('#'):
        continue
    f = i.rstrip().split('\t')
    black6 = f[3] + f[3]
    cast   = f[4] + f[4]
    hybrid = f[4] + f[3]
    # black6 is paternal; cast is maternal
    stdout.write('\t'.join([f[0], f[1], f[3], cast, black6, hybrid, 'PHASED']) + '\n')
