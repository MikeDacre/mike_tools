#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run scripts once a day if they haven't been finished yet

Writes last finished time and running pid to a json file,
if finished is less that 1 day submits a job.
"""
import os
import sys
import json
from datetime import datetime

sys.path.append(
    '/share/PI/hbfraser/modules/packages/anaconda3/5.0/lib/python3.6/site-packages'
)

import fyrd

__version__ = '0.1'

DATA_FILE = os.path.join(
    os.path.expandvars('$HOME'),
    '.fraser_lab_scripts.json'
)
PI_HOME = os.path.expandvars('$PI_HOME')

now = datetime.now()

if os.path.isfile(DATA_FILE):
    with open(DATA_FILE) as fin:
        last_job = json.load(fin)
else:
    last_job = None

# Decide if to run again
# Don't run is less than 1 day or if old jobs are running
if last_job:
    if (now-last_job['time']).days < 1:
        sys.exit(0)
    queue = fyrd.queue.Queue('self')
    open_jobs = [queue[i] for i in last_job['jobs'] if str(i) in queue.jobs]
    if open_jobs:
        for job in open_jobs:
            if job.state in fyrd.queue.ACTIVE_STATES:
                sys.exit(0)

# Get scripts
scripts = [
    os.path.join(PI_HOME, 'shared_environment', i) for i in os.listdir(PI_HOME)
]

# Create a temp dir just in case
TEMP = os.path.join(os.path.expandvars('$HOME'), '.tmp')
if not os.path.isdir(TEMP):
    os.makedirs(TEMP)

# Submit them
sub_jobs = []
for script in scripts:
    sub_jobs.append(
        fyrd.submit(
            'script', partition='hbfraser,hns,normal', cores=1, mem=4000,
            time='18:00:00', outfile='/dev/null', errfile='/dev/null',
            scriptpath=TEMP, runpath=TEMP, clean_files=True, clean_outputs=True
        )
    )

job_ids = []
for job in sub_jobs:
    job.update()
    job_ids.append(job.id)

# Write out data
job_data = {'time': now, 'jobs': job_ids}
with open(DATA_FILE, 'w') as fout:
    json.dump(job_data, fout)

# Done
sys.exit(0)
