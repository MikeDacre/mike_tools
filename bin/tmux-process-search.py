#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Search the tmux process tree.
"""
import sys as _sys
import argparse as _argparse

import subprocess as sub
import shlex as sh

import psutil

TMUX_LIST_CMD = 'tmux list-panes -a -F "#{pane_pid} #{session_name}" | grep {0}'


def find_session(pid):
    """Return the tmux session for a given pid."""
    proc = psutil.Process(pid)
    procs = []
    cur = proc
    while True:
        if cur.name() == 'tmux: server':
            if not procs:
                raise ValueError('You entered a tmux server PID')
            session_child = procs.pop()
            break
        procs.append(cur)
        cur = cur.parent()
        if cur is None:
            _sys.stderr.write(
                'PID {0} does not appear to be in a tmux session'
                .format(pid)
            )
            return None

    code, out, err = run(
		TMUX_LIST_CMD.format(session_child.pid), shell=True
	)

    if code != 0:
        _sys.stderr.write(
            'Failed to find tmux session of process\n'
        )
        return None
    return out.split(' ')[-1]



def run(cmd, shell=False, check=False, get='all'):
    """Replicate getstatusoutput from subprocess.

    Params
    ------
    cmd : str or list
    shell : bool, optional
        Run as a shell, allows piping
    check : bool, optional
        Raise exception if command failed
    get : {'all', 'code', 'stdout', 'stderr'}, optional
        Control what is returned:
            - all: (code, stdout, stderr)
            - code/stdout/stderr: only that item
            - None: code only

    Returns
    -------
    output : str or tuple
        See get above. Default return value: (code, stdout, stderr)
    """
    get_options = ['all', 'stdout', 'stderr', 'code', None]
    if get not in get_options:
        raise ValueError(
            'get must be one of {0} is {1}'.format(get_options, get)
        )
    if not shell and isinstance(cmd, str):
        cmd = sh.split(cmd)
    if get != 'code' and get is not None:
        pp = sub.Popen(cmd, shell=shell, stdout=sub.PIPE, stderr=sub.PIPE)
        out, err = pp.communicate()
    else:
        pp = sub.Popen(cmd, shell=shell)
        pp.communicate()
    if not isinstance(out, str):
        out = out.decode()
    if not isinstance(err, str):
        err = err.decode()
    code = pp.returncode
    if check and code != 0:
        if get:
            _sys.stderr.write(
                'Command failed\nSTDOUT:\n{0}\nSTDERR:\n{1}\n'
                .format(out, err)
            )
        raise sub.CalledProcessError(code, cmd)
    if get == 'all':
        return code, out.rstrip(), err.rstrip()
    elif get == 'stdout':
        return out.rstrip()
    elif get == 'stderr':
        return err.rstrip()
    return code


def main(argv=None):
    """Run as a script."""
    if not argv:
        argv = _sys.argv[1:]

    parser  = _argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('pid', type=int, help='PID to search for')

    args = parser.parse_args(argv)

    session_name = find_session(pid)


if __name__ == '__main__' and '__file__' in globals():
    _sys.exit(main())
