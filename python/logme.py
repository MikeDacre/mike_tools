"""
Logging with timestamps and optional log files.

================================================================================

          FILE: logme
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       CREATED: 2015-03-03 11:41
 Last modified: 2016-01-16 15:27

   DESCRIPTION: Print a timestamped message to a logfile, STDERR, or STDOUT.
                If STDERR or STDOUT are used, colored flags are added.
                Colored flags are INFO, WARNINING, ERROR, or CRITICAL.
                It is possible to wrtie to both logfile and STDOUT/STDERR
                using the also_write argument.

         USAGE: import logme as lm
                lm.log("Screw up!", <outfile>, kind='warn'|'error'|'normal',
                       also_write='stderr'|'stdout')

                All arguments are optional except for the initial message.

          NOTE: Uses terminal colors and STDERR, not compatible with non-unix
                systems

================================================================================
"""
import sys
import gzip
import bz2
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


def log(message, logfile=sys.stderr, kind='normal', also_write=None):
    """Print a string to logfile.

    :message: The message to print.
    :logfile: Optional file to log to, defaults to STDERR.
    :kind:    Prefix. Defaults to 'normal', options:
        'normal':   '<timestamp> INFO --> '
        'warn':     '<timestamp> WARNING --> '
        'class':    '<timestamp> ERROR --> '
        'critical': '<timestamp> CRITICAL --> '
    :also_write: 'stdout': print to STDOUT also.
    :also_write: 'stderr': print to STDERR also.
    """
    stdout = False
    stderr = False

    # Attempt to handle all file type and force append mode
    if isinstance(logfile, str):
        with _open_zipped(logfile, 'a') as outfile:
            _logit(message, outfile, kind, color=False)
    elif getattr(logfile, 'name').strip('<>') == 'stdout':
        _logit(message, logfile, kind, color=True)
        stdout = True
    elif getattr(logfile, 'name').strip('<>') == 'stderr':
        _logit(message, logfile, kind, color=True)
        stderr = True
    elif getattr(logfile, 'mode') == 'a':
        if getattr(logfile, 'closed'):
            with _open_zipped(logfile.name, 'a') as outfile:
                _logit(message, outfile, kind, color=False)
        else:
            _logit(message, outfile, kind, color=False)
    else:
        logfile.close()
        logfile = _open_zipped(logfile.name, 'a')
        _logit(message, outfile, kind, color=False)

    # Also print to stdout or stderr if requested
    if also_write == 'stdout' and not stdout:
        _logit(message, sys.stdout, kind, color=True)
    elif also_write == 'stderr' and not stderr:
        _logit(message, sys.stdout, kind, color=True)


###############################################################################
#                              Private Functions                              #
###############################################################################


def _logit(message, filehandle, kind, color=False):
    """Write message to file either with color or not."""
    if kind == 'normal':
        flag = 'INFO'
    if kind == 'warn':
        flag = 'WARNING'
    if kind == 'error':
        flag = 'ERROR'
    if kind == 'critical':
        flag = 'CRITICAL'

    if color:
        flag = _color(flag)

    now = dt.now()
    timestamp = "{}.{}".format(now.strftime("%Y%m%d %H:%M:%S"),
                               str(int(now.microsecond/1000)))
    filehandle.write('{0} | {1} --> {2}\n'.format(timestamp, flag,
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
