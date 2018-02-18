#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run scripts once a day if they haven't been finished yet

Writes last finished time and running pid to a json file, if finished is less
that 1 day submits a job.

This code forks a child process so that it does not slow down the calling
shell.

Code online here:
`https://github.com/MikeDacre/mike_tools/blob/master/bin/run_scripts.py`_
"""
import os
import sys
import json
import atexit
from datetime import datetime as dt

__version__ = '1.0-beta1'

##############
#  Contants  #
##############


LOCK_FILE = os.path.join(
    os.path.expandvars('$HOME'),
    '.fraser_lab_scripts.lock'
)
DATA_FILE = os.path.join(
    os.path.expandvars('$HOME'),
    '.fraser_lab_scripts.json'
)
PI_HOME = os.path.expandvars('$PI_HOME')
TEMP = os.path.join(os.path.expandvars('$HOME'), '.hbfraser-tmp')

PYTHON_LIB = os.path.join(PI_HOME, 'fyrd-isolated')

# Time to delay before rerunning in seconds (12 hours)
WAIT_TIME = 12*60*60


######################
#  Helper functions  #
######################


def check_pid(test_pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(test_pid, 0)
    except OSError:
        return False
    else:
        return True


###########################
#  Test if we should run  #
###########################

if os.path.isfile(LOCK_FILE):
    with open(LOCK_FILE) as infile:
        if check_pid(int(infile.read().strip())):
            sys.exit(0)


###################
#  Core Function  #
###################


def main():
    """Run core functionality."""

    # Our PID
    us = os.getpid()

    def exit_us(code=1):
        """Exit with code and delete lockfile if PID is us or dead."""
        if os.path.isfile(LOCK_FILE):
            with open(LOCK_FILE) as ffin:
                ppid = int(ffin.read().strip())
            if ppid == us or not check_pid(ppid):
                os.remove(LOCK_FILE)
        sys.exit(code)

    # Run exit_us at every exit (other than SIGKILL)
    atexit.register(exit_us)

    # Check we aren't already running
    if os.path.isfile(LOCK_FILE):
        with open(LOCK_FILE) as fin:
            pid = int(fin.read().strip())
        if check_pid(pid) and not pid == us:
            sys.exit(0)
        else:
            os.remove(LOCK_FILE)

    # Lock the script
    with open(LOCK_FILE, 'w') as fout:
        fout.write(str(us))

    # Time handling
    FMT = '%y%m%d-%H:%M:%S'
    NOW = dt.now()

    # Try to load old job data
    if os.path.isfile(DATA_FILE):
        with open(DATA_FILE) as fin:
            last_job = json.load(fin)
    else:
        last_job = None

    # Decide if we want to run again
    # Don't run is less than 12 hours or if old jobs are running
    if last_job:
        if (NOW-dt.strptime(last_job['time'], FMT)).seconds < WAIT_TIME:
            exit_us(0)

    # Only import fyrd when we have to as it can be slow
    sys.path.insert(0, PYTHON_LIB)
    import fyrd

    if last_job:
        queue = fyrd.queue.Queue('self')
        open_jobs = [queue[i] for i in last_job['jobs'] if str(i) in queue.jobs]
        if open_jobs:
            for job in open_jobs:
                if job.state in fyrd.queue.ACTIVE_STATES:
                    exit_us(0)

    # Create a temp dir just in case
    if not os.path.isdir(TEMP):
        os.makedirs(TEMP)

    # Clean up that dir
    fyrd.clean_dir(TEMP, confirm=False)
    os.system('rm {}/reset_perms* 2>/dev/null'.format(TEMP))
    os.system('rm {}/touch_pi_scratch* 2>/dev/null'.format(TEMP))

    # Get scripts
    SCRPT_PATH = os.path.join(PI_HOME, 'shared_environment')
    scripts = [
        os.path.join(SCRPT_PATH, i) for i in ['reset_perms.sh', 'touch_pi_scratch.sh']
    ]

    # Submit them
    sub_jobs = []
    for script in scripts:
        sub_jobs.append(
            fyrd.submit(
                'bash ' + script + ' >/dev/null 2>/dev/null',
                partition='hbfraser,hns,normal', cores=1, mem=4000,
                time='18:00:00', outfile='/dev/null', errfile='/dev/null',
                scriptpath=TEMP, runpath=TEMP, clean_files=True,
                clean_outputs=True, name=script.split('/')[-1].split('.')[0]
            )
        )

    # Convert jobs into ids only
    job_ids = []
    for job in sub_jobs:
        job.update()
        job_ids.append(job.id)

    # Write out data
    job_data = {'time': NOW.strftime(FMT), 'jobs': job_ids}
    with open(DATA_FILE, 'w') as fout:
        json.dump(job_data, fout)

    # Done, force delete file
    os.remove(LOCK_FILE)
    return 0


############
#  Script  #
############

# Fork this into the background
fpid = os.fork()

if (fpid == 0):
    os.setsid()
    fpid2 = os.fork()
    if (fpid2 == 0):
        # We are the second child, so we can do things
        sys.exit(main())
    else:
        sys.exit()
else:
    sys.exit()
