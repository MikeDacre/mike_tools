#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
"""
#====================================================================================
#
#          FILE: mike (python 3)
#        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
#  ORGANIZATION: Stanford University
#       LICENSE: Open Source - Public - Do as you wish (no license) - Mike Dacre
#       VERSION: 0.1
#       CREATED: 2013-08-26 10:39
# Last modified: 2013-08-26 14:28
#
#   DESCRIPTION: General functions that I use in my scripts
#
#         USAGE: Run as a script or import as a module.  See '-h' or 'help' for usage
#
#====================================================================================
"""

def open_log(logfile=''):
    """ Take either a string or a filehandle, detect if string.
        If string, open as file in append mode.
        If file, get name, close file, and reopen in append mode
        Return resulting file"""
    import sys

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
    """Print a string to logfile"""
    import sys

    output = str(output)

    stderr = False
    stdout = False

    if logfile:
        if isinstance(logfile, str):
            with open(logfile, 'a') as outfile:
                print(output, file=outfile)
        elif getattr(logfile, 'name') == '<stderr>':
            print(output, file=logfile)    
            stderr = True
        elif getattr(logfile, 'name') == '<stdout>':
            print(output, file=logfile)    
            stdout = True
        elif getattr(logfile, 'mode') == 'a':
            if getattr(logfile, 'closed'):
                with open(logfile.name, 'a') as outfile:
                    print(output, file=outfile)
            else:
                print(output, file=logfile)
        else:
            logfile.close()
            with open(logfile, 'a') as outfile:
                print(output, file=outfile)
    else:
        print(output, file=sys.stderr)
        stderr = True

    if print_level == 1 and not stdout:
        print(output)
    elif print_level == 2 and not stderr:
        print(output, file=sys.stderr)
