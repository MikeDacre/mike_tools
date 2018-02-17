"""
Functions to easily run shell commands.
"""
import os
import bz2
import gzip
import subprocess


def run(command, raise_on_error=False):
    """Run a command with subprocess the way it should be.

    Parameters
    ----------
    command : str
        A command to execute, piping is fine.
    raise_on_error : bool
        Raise a subprocess.CalledProcessError on exit_code != 0

    Returns
    -------
    stdout : str
    stderr : str
    exit_code : int
    """
    pp = subprocess.Popen(command, shell=True, universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = pp.communicate()
    code = pp.returncode
    if raise_on_error and code != 0:
        raise subprocess.CalledProcessError(
            returncode=code, cmd=command, output=out, stderr=err
        )
    return out, err, code


def open_zipped(infile, mode='r'):
    """Open a regular, gzipped, or bz2 file.

    If infile is a file handle or text device, it is returned without
    changes.

    Returns:
        text mode file handle.
    """
    mode   = mode[0] + 't'
    if hasattr(infile, 'write'):
        return infile
    if isinstance(infile, str):
        if infile.endswith('.gz'):
            return gzip.open(infile, mode)
        if infile.endswith('.bz2'):
            if hasattr(bz2, 'open'):
                return bz2.open(infile, mode)
            return bz2.BZ2File(infile, mode)
        return open(infile, mode)


def block_read(files, size=65536):
    """Iterate through a file by blocks."""
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(infile, force_blocks=False):
    """Return the line count of a file as quickly as possible.

    Uses `wc` if avaialable, otherwise does a rapid read.
    """
    if which('wc') and not force_blocks:
        if infile.endswith('.gz'):
            cat = 'zcat'
        elif infile.endswith('.bz2'):
            cat = 'bzcat'
        else:
            cat = 'cat'
        command = "{cat} {infile} | wc -l | awk '{{print $1}}'".format(
            cat=cat, infile=infile
        )
        return int(run(command)[0])
    else:
        with open_zipped(infile) as fin:
            return sum(bl.count("\n") for bl in block_read(fin))


def split_file(infile, parts, outpath='', keep_header=False):
    """Split a file in parts and return a list of paths.

    NOTE: Linux specific (uses wc).

    **Note**: If has_header is True, the top line is stripped off the infile
    prior to splitting and assumed to be the header.

    Args:
        outpath:     The directory to save the split files.
        has_header:  Add the header line to the top of every file.

    Returns:
        list: Paths to split files.
    """
    # Determine how many reads will be in each split sam file.

    num_lines = int(count_lines(infile)/int(parts)) + 1

    # Subset the file into X number of jobs, maintain extension
    cnt       = 0
    currjob   = 1
    suffix    = '.split_' + str(currjob).zfill(4) + '.' + infile.split('.')[-1]
    file_name = os.path.basename(infile)
    run_file  = os.path.join(outpath, file_name + suffix)
    outfiles  = [run_file]

    # Actually split the file
    with open_zipped(infile) as fin:
        header = fin.readline() if keep_header else ''
        sfile = open_zipped(run_file, 'w')
        sfile.write(header)
        for line in fin:
            cnt += 1
            if cnt < num_lines:
                sfile.write(line)
            elif cnt == num_lines:
                sfile.write(line)
                sfile.close()
                currjob += 1
                suffix = '.split_' + str(currjob).zfill(4) + '.' + \
                    infile.split('.')[-1]
                run_file = os.path.join(outpath, file_name + suffix)
                sfile = open_zipped(run_file, 'w')
                outfiles.append(run_file)
                sfile.write(header)
                cnt = 0
        sfile.close()
    return tuple(outfiles)


def is_exe(fpath):
    """Return True is fpath is executable."""
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def file_type(infile):
    """Return file type after stripping gz or bz2."""
    name_parts = infile.split('.')
    if name_parts[-1] == 'gz' or name_parts[-1] == 'bz2':
        name_parts.pop()
    return name_parts[-1]


def is_file_type(infile, types):
    """Return True if infile is one of types.

    Args:
        infile:  Any file name
        types:   String or list/tuple of strings (e.g ['bed', 'gtf'])

    Returns:
        True or False
    """
    if hasattr(infile, 'write'):
        infile = infile.name
    types = listify(types)
    for typ in types:
        if file_type(infile) == typ:
            return True
    return False


def which(program):
    """Replicate the UNIX which command.

    Taken verbatim from:
        stackoverflow.com/questions/377017/test-if-executable-exists-in-python

    Args:
        program: Name of executable to test.

    Returns:
        Path to the program or None on failu_re.
    """
    fpath, program = os.path.split(program)
    if fpath:
        if is_exe(program):
            return os.path.abspath(program)
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return os.path.abspath(exe_file)

    return None


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def listify(iterable):
    """Try to force any iterable into a list sensibly."""
    if isinstance(iterable, list):
        return iterable
    if isinstance(iterable, (str, int, float)):
        return [iterable]
    if not iterable:
        return []
    if callable(iterable):
        iterable = iterable()
    return list(iter(iterable))
