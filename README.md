Mike's Random Scripts
=====================

A selection of little unrelated tools that do not merit their own repos.  Use as you wish.

The tools in python/ are intended to be placed in python's runtime path for use by import. The tools in bin/ are intended to be run directly.

Pipeline
--------
In the python directory, the pipeline.py file contains functions to build and manage a complete pipeline with python2 or python3. Full documentation is in that file.

It allows the user to build a pipeline by step using any executable, shell script, or python function as a step. It also supports adding a python function to test for failure. Once all steps have been added, the run_all() function can be used to execute them in order, execution will terminate if a step fails.

The pipeline object is autosaved using pickle, so no work is lost on any failure (except if the managing script dies during the execution of a step).

All STDOUT, STDERR, return values, and exit codes are saved by default, as are exact start and end times for every step, making future debugging easy. Steps can be rerun at any time. run_all() automatically starts from the last completed step, unless explicitly told to start from the beginning.

Failure tests can be directly called also, allowing the user to set a step as done, even if the parent script died during execution.

In the future this will be extended to work with slurmy, right now no steps can be run with job managers, as the job submission will end successfully before the step has completed, breaking dependency tracking.

logme
-----
In the python directory, the logme.py file allows easy timestamped and color coded logging to virtually any file type or logging object.
