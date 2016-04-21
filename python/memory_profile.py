"""
Simple profiler for memory usage of a script.

============================================================================

   ORIG AUTHOR: Unknown
        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
       CREATED: 2016-44-07 11:03
 Last modified: 2016-03-07 11:47

   DESCRIPTION: Only works on *nix OSes

============================================================================
"""
import os
_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '''Return memory usage in megabytes.
    '''
    return int((_VmB('VmSize:') - since)/1024/1024)


def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since
