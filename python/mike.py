#!/usr/bin/env python3
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
# Last modified: 2015-03-03 11:45
#
#   DESCRIPTION: General functions that I use in my scripts
#
#         USAGE: For importing.
#
#====================================================================================
"""
import logme, sys


def pbs_submit(command, name='', template=''):
    """Take a command and an optional template and submit it to PBS. Return job number"""

    if not template:
        template = """#!/bin/bash
#PBS -S /bin/bash
#PBS -m ae
"""
    if not name:
        name = command

    template = '\n'.join([template, ' '.join(["#PBS -N", name]), "cd $PBS_O_WORKDIR", ''])

    # Open pbs session
    pbs_command = subprocess.Popen('qsub', stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    # Submit command
    pbs_command.stdin.write('\n'.join([template, command]).encode())
    pbs_command.stdin.close()

    # Get job number
    output = pbs_command.stdout.read().decode().rstrip()
    pbs_command.stdout.close()

    return output

def qstat_mon(job_list, verbose=False, logfile=sys.stderr):
    """ Take a list of job numbers and monitor them for completion.
        List can be either a plain list of a dictionary with
        { name : job_number } format, where name is any name you
        wish
        Returns a tuple with success/failure and dictionary of exit codes
        If an exit code is not positive, will return failure as false
    """
    import subprocess, re
    from time import sleep

    # Make a dictionary
    final_list = {}
    if isinstance(job_list, list) or isinstance(job_list, tuple):
        for i in job_list:
            final_list[str(i)] = i
    elif isinstance(job_list, dict):
        final_list = job_list
    else:
        logme("qstat_mon: Job list is not a valid list or dictionary", logfile, 2)
        return False, final_list

    if verbose:
        logme(' '.join(["Monitoring", str(len(final_list)), "jobs"]), logfile, 2)
    else:
        logme(' '.join(["Monitoring", str(len(final_list)), "jobs"]), logfile)

    # Check jobs, raise exception if any fail, otherwise we are done
    complete_jobs = {}
    while 1:
        for name, job_number in final_list.items():
            if name not in complete_jobs.keys():
                if re.search(r' C ', subprocess.check_output(['qstat', str(job_number)]).decode().rstrip()):
                    exit_code = re.findall(r'exit_status = ([0-9]+)', subprocess.check_output(['qstat', '-f', str(job_number)]).decode())[0]
                    complete_jobs[name] = exit_code
                    if verbose:
                        logme(' '.join([name, "completed"]), logfile, 2)
        if len(complete_jobs) == len(final_list):
            break
        else:
            sleep(5)

    if verbose:
        logme("qstat_mon: All submitted jobs complete", logfile, 2)
    else:
        logme("qstat_mon: All submitted jobs complete", logfile)

    # Check for errors
    error_jobs = []
    success = True
    for name, exit_code in complete_jobs.items():
        if int(exit_code):
            success = False
            error_jobs.append(name)

    # Log info on error jobs if they exist
    if error_jobs:
        logme("qstat_mon: One or more jobs failed!", logfile, 2)
        for name, exit_code in complete_jobs.items():
            if name in error_jobs:
                logme(' '.join([name, "finished with exit code", exit_code]), logfile)

    # Return dictionary of completed jobs
    return success, complete_jobs

# If executed directly, inform the user that they are a little silly.
if __name__ == '__main__':
    print("This script is a function bucket, don't run as a script")
