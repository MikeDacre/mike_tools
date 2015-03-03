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
#          FILE: logme (python 3)
#        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
#  ORGANIZATION: Stanford University
#       CREATED: 2015-03-03 11:41
# Last modified: 2015-03-03 11:45
#
#   DESCRIPTION: Print a string to a logfile. logme can print to STDOUT or
#                STDERR also:
#                print_level: 1: print to STDOUT also
#                print_level: 2: print to STDERR also
#
#         USAGE: import logme
#                logme("Screw up!", sys.stderr, print_level=2
#
#====================================================================================
"""
import sys

def open_log(logfile=''):
    """ Take either a string or a filehandle, detect if string.
        If string, open as file in append mode.
        If file, get name, close file, and reopen in append mode
        Return resulting file"""

    if logfile:
        if isinstance(logfile, str):
            finalfile = open(logfile, 'a')
        elif getattr(logfile, 'name') == '<stderr>' or getattr(logfile, 'name') == '<stdout>':
            finalfile = logfile
        elif not getattr(logfile, 'mode') == 'a':
            logfile.close()
            finalfile = open(logfile, 'a')
        else:
            raise Exception("logfile is not valid")
    else:
        finalfile = sys.stderr

    return finalfile

def logme(output, logfile='', print_level=0):
    """ Print a string to logfile
        print_level: 1: print to STDOUT also
        print_level: 2: print to STDERR also """
    import datetime

    timestamp   = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
    output      = str(output)
    timeput     = ' | '.join([timestamp, output])

    stderr = False
    stdout = False

    if logfile:
        if isinstance(logfile, str):
            with open(logfile, 'a') as outfile:
                print(timeput, file=outfile)
        elif getattr(logfile, 'name') == '<stderr>':
            print(timeput, file=logfile)
            stderr = True
        elif getattr(logfile, 'name') == '<stdout>':
            print(timeput, file=logfile)
            stdout = True
        elif getattr(logfile, 'mode') == 'a':
            if getattr(logfile, 'closed'):
                with open(logfile.name, 'a') as outfile:
                    print(timeput, file=outfile)
            else:
                print(timeput, file=logfile)
        else:
            logfile.close()
            with open(logfile, 'a') as outfile:
                print(timeput, file=outfile)
    else:
        print(timeput, file=sys.stderr)
        stderr = True

    if print_level == 1 and not stdout:
        print(output)
    elif print_level == 2 and not stderr:
        print(output, file=sys.stderr)
