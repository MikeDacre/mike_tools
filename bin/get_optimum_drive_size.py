#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Distributed under terms of the MIT license - aka free, open source, use as you wish
"""
Provide the script with a block device id (e.g. sda or sdc) and it
will return an integer that can be used as the optimum starting point in
sectors for a partion.

Use with parted as such:
    mkpart <partition_name> <output_of_this_script> 100%

The result will be a well formatted partition that occupies 100% of the disk space

Based on the excellent tip by Ian Chard (@Flupsybunny) at:
    <http://rainbow.chard.org/2013/01/30/how-to-align-partitions-for-best-performance-using-parted/>
"""

from sys import argv, exit

if len(argv)<2 or argv[1]=='-h' or argv[1]=='--help':
    print(__doc__)
    exit(1)

drive=argv[1]

try:
    optimal_io_size = int(open('/sys/block/' + drive + '/queue/optimal_io_size').read().rstrip())
    alignment_offset = int(open('/sys/block/' + drive + '/alignment_offset').read().rstrip())
    physical_block_size = int(open('/sys/block/' + drive + '/queue/physical_block_size').read().rstrip())
except FileNotFoundError:
    print("Device", drive, "not found")
    exit(2)

result = int((optimal_io_size + alignment_offset) / physical_block_size)

print(str(result))
