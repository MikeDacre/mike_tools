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
# Last modified: 2014-08-22 16:56
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
#                -v or --variables is a list of variables to pass to the function       #
#                    This list must be space separated. Each space separated item will  #
#                    be run as a separate matlab job.                                   #
#                    You may comma separate variables if you wish those variables to    #
#                    passed as multiple arguments to a single function (e.g. not as     #
#                    separate jobs.                                                     #
#                                                                                       #
#                <function> is a positional arg and is the name of the function to run  #
#                                                                                       #
#                Run as a script or import as a module.  See '-h' or 'help' for usage   #
#                                                                                       #
#=======================================================================================#
"""
import logme
from pbs_torque import job

def _get_args():
    """Command Line Argument Parsing"""
    import argparse, sys

    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Optional Arguments
    parser.add_argument('-p', '--path', nargs='?',
                        help="Comma or space separated list of matlab paths")
    parser.add_argument('-V', '--variables', nargs='?',
                        help="Space separated list variables to pass to function")
    parser.add_argument('-v', dest='verbose', help="Verbose output")

    # Function name
    parser.add_argument('function',
                        help="Name of the matlab function to run")

    # Optional Files
    parser.add_argument('-l', '--logfile', nargs='?',
                        help="Log File, Default STDERR (append mode)")

    return parser

# Main function for direct running
def main():
    """Run directly"""
    # Get commandline arguments
    parser = _get_args()
    args = parser.parse_args()


# The end
if __name__ == '__main__':
    main()
