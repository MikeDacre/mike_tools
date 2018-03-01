#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execute rm, but be careful about it.
"""
import os
import sys
from glob import glob
from getpass import getuser
from builtins import input
from subprocess import call

# Don't ask if fewer than this number of files deleted
CUTOFF = 2

# Where to move files to if recycled
RECYCLE_BIN = os.path.expandvars('/tmp/{}_trash'.format(getuser()))


def yesno(message, def_yes=True):
    """Get a yes or no answer from the user."""
    message += ' [Y/n] ' if def_yes else ' [y/N] '
    ans = input(message)
    if ans:
        return ans.lower() == 'y' or ans.lower() == 'yes'
    return def_yes


def main(argv=sys.argv):
    """Safe rm"""
    if not argv:
        sys.stderr.write(
            'Arguments required\n'
            'Usage: rm [--recycle|-c] [options] files\b'
        )
        return 99
    flags = []
    all_files = []
    recursive = False
    recycle = False
    for arg in sys.argv[1:]:
        if arg == '--recycle' or arg == '-c':
            recycle = True
        elif arg.startswith('-'):
            if 'r' in arg:
                recursive = True
            flags.append(arg)
        else:
            all_files += glob(arg)
    if recycle:
        sys.stderr.write(
            'All files will be recycled to {}\n'.format(RECYCLE_BIN)
        )
    drs = []
    fls = []
    bad = []
    for fl in all_files:
        if os.path.isdir(fl):
            drs.append(fl)
        elif os.path.isfile(fl):
            fls.append(fl)
        else:
            bad.append(fl)
    if bad:
        sys.stderr.write(
            'The following files do not match any files\n{}\n'
            .format(' '.join(bad))
        )
    ld = len(drs)
    if recursive:
        if drs:
            dc = 0
            fc = 0
            for dr in drs:
                for i in [os.path.join(dr, d) for d in os.listdir(dr)]:
                    if os.path.isdir(i):
                        dc += 1
                    else:
                        fc += 1
            if dc or fc:
                info = []
                if fc:
                    info.append('{} subfiles'.format(fc))
                if dc:
                    info.append('{} subfolders'.format(dc))
            inf = ' and '.join(info)
            msg = 'Recursively deleting '
            if ld < 6:
                msg += 'the folders {}'.format(drs)
                if info:
                    msg += ' with ' + inf
            else:
                msg += '{} dirs:'.format(ld)
                msg += '\n{}\n'.format(' '.join(drs))
                if info:
                    msg += 'Containing ' + inf
                else:
                    msg += 'Containing no subfiles or directories'
            sys.stderr.write(msg + '\n')
            if not yesno('Really delete?', False):
                return 1
    elif drs:
        if ld < 6:
            sys.stderr.write(
                'Directories {} included but -r not sent\n'
                .format(drs)
            )
        else:
            sys.stderr.write(
                '{} directories included but -r not sent\n'
                .format(len(drs))
            )
        if not yesno('Continue anyway?'):
            return 2
        drs = []
    if len(fls) >= CUTOFF:
        if len(fls) < 6:
            if not yesno('Delete the files {}?'.format(fls), False):
                return 6
        else:
            sys.stderr.write(
                'Deleting the following {} files:\n{}\n'
                .format(len(fls), ' '.join(fls))
            )
            if not yesno('Delete?', False):
                return 10
    to_delete = drs + fls
    to_delete = ['"' + i + '"' for i in to_delete]
    if not to_delete:
        sys.stderr.write('No files or folders to delete\n')
        return 22
    if recycle:
        if not os.path.isdir(RECYCLE_BIN):
            os.makedirs(RECYCLE_BIN)
        return call(
            'mv -i {} {}'.format(' '.join(to_delete), RECYCLE_BIN),
            shell=True
        )
    return call(
        'rm {}'.format(' '.join(flags + to_delete)),
        shell=True
    )


if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
