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
# Last modified: 2013-08-26 11:38
#
#   DESCRIPTION: General functions that I use in my scripts
#
#         USAGE: Run as a script or import as a module.  See '-h' or 'help' for usage
#
#====================================================================================
"""

def logme(logfile):
    """ Take either a string or a filehandle, detect if string.
        If string, open as file in append mode.
        If file, get name, close file, and reopen in append mode
        Return resulting file"""
    if isinstance(logfile, str):
        finalfile = open(logfile, 'a')
    elif getattr(logfile, 'name') == '<stderr>' or getattr(logfile, 'name') == '<stdout>':
        finalfile = logfile
    elif not getattr(logfile, 'mode') == 'a':
        logfile.close()
        finalfile = open(logfile, 'a')
    else:
        raise Exception "logfile is not valid"

    return logfile

