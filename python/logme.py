#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:set et sw=4 ts=4 tw=80:
"""
================================================================================

          FILE: logme
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       CREATED: 2015-03-03 11:41
 Last modified: 2016-01-08 15:51

   DESCRIPTION: Print a string to a logfile. logme can print to STDOUT or
                STDERR also:
                print_level: 1: print to STDOUT also
                print_level: 2: print to STDERR also

         USAGE: import logme
                logme("Screw up!", sys.stderr, print_level=2

================================================================================
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
        elif getattr(logfile, 'name') == '<stderr>' \
                or getattr(logfile, 'name') == '<stdout>':
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
                outfile.write(timeput + '\n')
        elif getattr(logfile, 'name') == '<stderr>':
            logfile.write(timeput + '\n')
            stderr = True
        elif getattr(logfile, 'name') == '<stdout>':
            logfile.write(timeput + '\n')
            stdout = True
        elif getattr(logfile, 'mode') == 'a':
            if getattr(logfile, 'closed'):
                with open(logfile.name, 'a') as outfile:
                    outfile.write(timeput + '\n')
            else:
                logfile.write(timeput + '\n')
        else:
            logfile.close()
            with open(logfile, 'a') as outfile:
                outfile.write(timeput + '\n')
    else:
        sys.stderr.write(timeput + '\n')
        stderr = True

    if print_level == 1 and not stdout:
        sys.stdout.write(output + '\n')
    elif print_level == 2 and not stderr:
        sys.stderr.write(output + '\n')
