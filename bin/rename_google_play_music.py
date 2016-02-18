#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rename mp3s as '[track] [name].mp3'.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
       CREATED: 2016-37-17 16:02
 Last modified: 2016-02-17 16:50

============================================================================
"""
import os
import sys
import argparse
import mutagen.mp3


def rename_files(files):
    """Rename files using mp3 tags.

    :files:   List of files to run on
    :returns: 0 on success

    """
    for fl in files:
        metadata = mutagen.mp3.Open(fl)
        os.rename(fl, '{0:0>2} {1}.mp3'.format(
            str(metadata['TRCK']).split('/')[0], metadata['TIT2']))


def main(argv=None):
    """ Run as a script """
    if not argv:
        argv = sys.argv[1:]

    parser  = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('files', nargs='+',
                        help="Input files")

    args = parser.parse_args(argv)

    rename_files(args.files)

if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
