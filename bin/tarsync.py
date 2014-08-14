#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Copyright © Mike Dacre <mike.dacre@gmail.com>
#
# Distributed under terms of the MIT license
"""
#====================================================================================
#
#          FILE: tarsync (python 3)
#        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
#  ORGANIZATION: Stanford University
#       LICENSE: MIT License, Property of Stanford, Use as you wish
#       VERSION: 1.0 (beta)
#       CREATED: 2014-08-13 12:34
# Last modified: 2014-08-13 16:30
#
#   DESCRIPTION: rsync is fantastic for incremental backups, but it is really
#                slow for initial transfers where large amounts of data have to
#                be copied. tar on the other hand can easily reach 100% IO
#                usage on an external USB3 drive - much, much faster - but yet
#                it has no ability to detect incremental changes
#
#                tarsync addresses this issue, by using rsync to build a list
#                of files that need to be transfered, and then using tar to do
#                the actual transfer.
#
#         USAGE: Run as a script or import as a module.  See '-h' or 'help' for usage
#
#====================================================================================
"""

def tarsync(start, end, verbose=False):
    """ Use rsync comparison to tar copy files """
    from subprocess import call, check_output, Popen, PIPE
    from os import path, chdir, makedirs
    from re import sub
    from sys import stderr, exit

    # Error checking
    if not path.exists(start):
        print(_c('red', "Copy from location does not exist", bold=True), file=stderr)
        exit(1)
    if not path.isdir(start):
        print(_c('red', "tarsync only works on directories", bold=True), file=stderr)
        exit(2)

    # Process paths
    initial_dir = path.abspath(path.curdir)
    start    = path.abspath(start)
    end      = path.abspath(end)
    rstart   = start + '/'
    rend     = end + '/'

    j = check_output(['rsync', '--verbose', '--recursive', '--dry-run', rstart, rend]).decode('utf8').rstrip().split('\n')[1:-3]
    j = [i for i in filter(None, j)]
    l = len(j)

    if l == 0:
        print(_c('green', "Folders are the same", bold=True), file=stderr)
        exit(0)

    if verbose:
        print(_c('green', "Rsync File List:"), file=stderr)
        print('\n'.join(j), file=stderr)

    if not path.exists(end):
        print(_c('green', "\nDestination folder does not exist"), file=stderr)
        print(_c('green', "Doing direct tar copy"), file=stderr)

        makedirs(end)

        chdir(start)
        tar_send = Popen(['tar', 'cf', '-', '.'], stdout = PIPE)
        chdir(end)
        Popen(['tar', 'xf', '-'], stdin = tar_send.stdout)
        chdir(initial_dir)

    else:
        for i in j:
            i = sub(start + '/', '', i)
            if not i:
                continue

            if verbose:
                print(_c('cyan', "Copying " + i), file=stderr)

            if i.endswith('/'):
                if not path.exists(end + '/' + i):
                    makedirs(end + '/' + i)
            else:
                chdir(start)
                tar_send = Popen(['tar', 'cf', '-', i], stdout = PIPE)
                chdir(end)
                Popen(['tar', 'xf', '-'], stdin = tar_send.stdout)
                chdir(initial_dir)

    # Final rsync confirmation
    commands = ['rsync', '--verbose', '--recursive', rstart, rend]
    if verbose:
        print("\n" + _c('green', "Running final rsync check:"), file=stderr)
        call(commands)
    else:
        check_output(commands)

    print(_c('green', "\nFiles/directories transfered: " + str(l), bold=True), file=stderr)

def _c(color, string, bold=False):
    """ Print in color """
    # Set colors
    if color == 'red':
        color = '\033[31'
    elif color == 'green':
        color = '\033[32'
    elif color == 'yellow':
        color = '\033[32'
    elif color == 'blue':
        color = '\033[32'
    elif color == 'magenta':
        color = '\033[32'
    elif color == 'cyan':
        color = '\033[32'
    else:
        return(string)

    color = color + ";01m" if bold else color + "m"

    return(color + string + '\033[0m')

def _get_args():
    """Command Line Argument Parsing"""
    import argparse, sys

    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional Arguments
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help="Verbose output")

    # Required Arguments
    parser.add_argument('copy_from',
                        help="Directory to copy")
    parser.add_argument('copy_to',
                        help="Location to copy to")

    return parser

# Main function for direct running
def main():
    """Run directly"""
    # Get commandline arguments
    parser = _get_args()
    args = parser.parse_args()

    tarsync(args.copy_from, args.copy_to, args.verbose)

# The end
if __name__ == '__main__':
    main()
