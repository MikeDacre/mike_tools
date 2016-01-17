"""
Logging with timestamps and optional log files.

================================================================================

          FILE: logme
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       CREATED: 2015-03-03 11:41
 Last modified: 2016-01-17 09:06

   DESCRIPTION: Print a timestamped message to a logfile, STDERR, or STDOUT.
                If STDERR or STDOUT are used, colored flags are added.
                Colored flags are INFO, WARNINING, ERROR, or CRITICAL.
                It is possible to wrtie to both logfile and STDOUT/STDERR
                using the also_write argument.
                'level' can also be provided, logs will only print if
                level > kind. critical<error<warn<info<debug

         USAGE: import logme as lm
                lm.log("Screw up!", <outfile>, kind='warn'|'error'|'normal',
                       also_write='stderr'|'stdout', level='error')

                All arguments are optional except for the initial message.

          NOTE: Uses terminal colors and STDERR, not compatible with non-unix
                systems

================================================================================
"""
import sys
import gzip
import bz2
import logging
from datetime import datetime as dt

__all__ = ['log']

###################################
#  Constants for printing colors  #
###################################

WHITE  = '\033[97m'
YELLOW = '\033[93m'
RED    = '\033[91m'
BOLD   = '\033[1m'
ENDC   = '\033[0m'


def log(message, logfile=sys.stderr, kind='info', also_write=None, level=None):
    """Print a string to logfile.

    :message: The message to print.
    :logfile: Optional file to log to, defaults to STDERR. Can provide a
              logging object
    :kind:    Prefix. Defaults to 'normal', options:
        'debug':    '<timestamp> DEBUG --> '
        'info':     '<timestamp> INFO --> '
        'warn':     '<timestamp> WARNING --> '
        'error':    '<timestamp> ERROR --> '
        'critical': '<timestamp> CRITICAL --> '
    :also_write: 'stdout': print to STDOUT also.
    :also_write: 'stderr': print to STDERR also.

    :level: The minimum print level, same flags as 'kind', must be greater
            than 'kind' to print.
    """
    stdout = False
    stderr = False
    message = str(message)

    # Attempt to handle all file type
    if isinstance(logfile, (logging.RootLogger, logging.Logger)):
        _logit(message, logfile, kind, color=False, level=level)
    elif isinstance(logfile, str):
        with _open_zipped(logfile, 'a') as outfile:
            _logit(message, outfile, kind, color=False, level=level)
    elif str(getattr(logfile, 'name')).strip('<>') == 'stdout':
        _logit(message, logfile, kind, color=True, level=level)
        stdout = True
    elif str(getattr(logfile, 'name')).strip('<>') == 'stderr':
        _logit(message, logfile, kind, color=True, level=level)
        stderr = True
    elif getattr(logfile, 'closed'):
        with _open_zipped(logfile.name, 'a') as outfile:
            _logit(message, outfile, kind, color=False, level=level)
    else:
        _logit(message, logfile, kind, color=False, level=level)

    # Also print to stdout or stderr if requested
    if also_write == 'stdout' and not stdout:
        _logit(message, sys.stdout, kind, color=True, level=level)
    elif also_write == 'stderr' and not stderr:
        _logit(message, sys.stdout, kind, color=True, level=level)

def clear(infile):
    """Truncate a file."""
    open(infile, 'w').close()


###############################################################################
#                              Private Functions                              #
###############################################################################


def _logit(message, output, kind, color=False, level=None):
    """Write message to file either with color or not.

    output must be filehandle or logging object.
    """
    # Level checking, not used with logging objects
    level_map = {'debug': 0, 'info': 1, 'warn': 2, 'error': 3, 'critical': 4,
                 'd': 0, 'i': 1, 'w': 2, 'e': 3, 'c': 4,
                 0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    flag_map  = {0: 'DEBUG', 1: 'INFO', 2: 'WARNING', 3: 'ERROR',
                 4: 'CRITICAL'}

    now = dt.now()
    timestamp = "{0}.{1:<3}".format(now.strftime("%Y%m%d %H:%M:%S"),
                                    str(int(now.microsecond/1000)))

    try:
        kind = level_map[kind]
    except KeyError:
        raise Exception('Invalid kind {}'.format(kind))

    flag = flag_map[kind]
    flag_len = len('{0} | {1} --> '.format(timestamp, flag)) - 2

    if color:
        flag = _color(flag)

    if isinstance(output, (logging.RootLogger, logging.Logger)):
        message = ' {} --> {}'.format(timestamp, message)
        if kind == 0:
            output.debug(message)
        if kind == 1:
            output.info(message)
        if kind == 2:
            output.warning(message)
        if kind == 3:
            output.error(message)
        if kind == 4:
            output.critical(message)
    else:
        # Check level before proceeding
        if kind < level_map[level]:
            return

        # Format multiline message
        lines = message.split('\n')
        if len(lines) != 1:
            message = lines[0] + '\n'
            lines = lines[1:]
            for line in lines:
                message = message + ''.ljust(flag_len, '-') + '> ' + line + '\n'
        output.write('{0} | {1} --> {2}\n'.format(timestamp, flag,
                                                      str(message)))


def _color(flag):
    """Return the flag with correct color codes."""
    if flag == 'INFO':
        return BOLD + WHITE + flag + ENDC
    if flag == 'WARNING':
        return BOLD + YELLOW + flag + ENDC
    if flag == 'ERROR':
        return BOLD + RED + flag + ENDC
    if flag == 'CRITICAL':
        return BOLD + RED + flag + ENDC


def _open_zipped(infile, mode='r'):
    """Return file handle of file regardless of zipped or not.

    Text mode enforced for compatibility with python2
    """
    mode   = mode[0] + 't'
    p2mode = mode
    if hasattr(infile, 'write'):
        return infile
    if isinstance(infile, str):
        if infile.endswith('.gz'):
            return gzip.open(infile, mode)
        if infile.endswith('.bz2'):
            if hasattr(bz2, 'open'):
                return bz2.open(infile, mode)
            else:
                return bz2.BZ2File(infile, p2mode)
        return open(infile, p2mode)
