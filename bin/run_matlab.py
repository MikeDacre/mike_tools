#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Copyright © Mike Dacre <mike.dacre@gmail.com>
#
# Distributed under terms of the MIT license
"""
#=======================================================================================#
#                                                                                       #
#          FILE: run_matlab (python 3)                                                  #
#        AUTHOR: Michael D Dacre, mike.dacre@gmail.com                                  #
#  ORGANIZATION: Stanford University                                                    #
#       LICENSE: MIT License, Property of Stanford, Use as you wish                     #
#       VERSION: 0.1                                                                    #
#       CREATED: 2014-08-22 16:26                                                       #
# Last modified: 2014-08-25 13:23
#                                                                                       #
#   DESCRIPTION: Create a bunch of temporary matlab scripts to call some other          #
#                matlab script and then submit to the cluster.                          #
#                                                                                       #
#                Requires that the matlab function be written to accept imput           #
#                variables.                                                             #
#                                                                                       #
#                Right now only works with torque jobs tools and requires               #
#                pbs_torque from                                                        #
#                https://github.com/MikeDacre/torque_queue_manager and logging          #
#                functions from http://j.mp/python_logme                                #
#                                                                                       #
#         USAGE: -p or --path allows the addition of multiple matlab paths              #
#                <function> is a positional arg and is the name of the function to run  #
#                                                                                       #
#                STDIN: Variable list for matlab                                        #
#                    This list must be space or newline separated. Each space separated #
#                    item will be run as a separate matlab job.                         #         #
#                    To provide multiple variables to the matlab function,              #
#                    comma separate the variables on a single line.                     #
#                                                                                       #
#                                                                                       #
#                Run as a script or import as a module.  See '-h' or 'help' for usage   #
#                                                                                       #
#=======================================================================================#
"""
# Set to true to get pointlessly verbose output
debug=True

def submit_matlab_jobs(paths, variables, function, verbose=False, logfile=None):
    """ Take a list of paths, a list of lists of variables, and a single Function
        and submit one job for every list in the lists of variables (each item of
        the second dimension will be submitted as an additional argument to the
        matlab function. """
    import logme
    from pbs_torque import job

def _get_args():
    """ Command Line Argument Parsing """
    import argparse

    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional Arguments
    parser.add_argument('-p', '--path',
                        help="Comma separated list of matlab paths")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose output")

    # Function name
    parser.add_argument('function',
                        help="Name of the matlab function to run")

    # Optional Files
    parser.add_argument('-l', '--logfile',
                        help="Log File, Default STDERR (append mode)")

    return parser

# Main function for direct running
def main():
    """ Run directly """
    from sys import stdin
    from os import path

    # Get commandline arguments
    parser = _get_args()
    args = parser.parse_args()

    # Get variable list from STDIN
    variables = [i.split(',') for i in stdin.read().rstrip().split('\n')]

    # Split paths
    paths = args.path.split(',')

    if debug:
        print(paths)
        print(args.function)
        print(variables)

    submit_matlab_jobs(paths=paths, variables=variables, function=function, verbose=args.verbose, logfile=args.logfile)

# The end
if __name__ == '__main__':
    main()
