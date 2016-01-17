import os
import sys
import pipeline as pl
import pytest
from logme import log

PIPELINE_FILE = 'test.test'
LOGFILE = 'test_pipeline.log'
MIN_LEVEL = 'info'

pipeline_output = ''

###############################################################################
#                           Non-Test Functions                                #
###############################################################################

def lg(msg, level):
    """Run logme log with LOGFILE"""
    log(msg, LOGFILE, level=level, min_level=MIN_LEVEL)


def write_something(msg):
    """Write msg to stdout"""
    return(msg + '\n')


def get_pipeline():
    """Get the pipeline object."""
    return  pl.get_pipeline(PIPELINE_FILE)


def create_pipeline():
    """Create pipeline"""
    pl.get_pipeline(PIPELINE_FILE)


def remove_pipeline():
    """Delete pipeline file"""
    if os.path.exists(PIPELINE_FILE):
        os.remove(PIPELINE_FILE)


###############################################################################
#                               Test Functions                                #
###############################################################################


def test_pipeline_creation():
    """Create a Pipeline object and add a few steps."""
    remove_pipeline()
    create_pipeline()


def test_additions():
    """Add methods to pipeline."""
    pip = get_pipeline()
    pip.add('ls')
    pip.add('ls', '~', name='bob')
    pip.add_function(write_something, 'call1')
    pip.add(write_something, args=('call2',), name='run2')
    pip.add('ls sldfkj', name='tofail')
    assert len(pip) == 5
    pip.save()
    global pipeline_output
    pipeline_output = str(pip)
    lg(str(get_pipeline()), level=0)


def test_restore():
    """Make sure methods still exist"""
    pip = get_pipeline()
    assert len(pip) == 5
    assert pip['bob'].command == pl.get_path('ls')
    assert pip['bob'].args == '~'
    assert pip['bob'].done == False
    assert isinstance(pip['bob'], pl.Command)
    assert isinstance(pip['run2'], pl.Function)
    assert isinstance(pip['tofail'], pl.Command)
    assert str(pip['write_something'].command).startswith(
        '<function write_something')
    assert pip['run2'].args == ('call2',)
    assert str(pip) == pipeline_output
    lg(str(get_pipeline()), level=0)


def test_run():
    """Execute pipeline steps."""
    pip = get_pipeline()
    with pytest.raises(pl.Command.CommandFailed):
        pip.run_all()
    for step in list(pip)[:4]:
        assert step.done == True
    assert pip['tofail'].done == False
    lg('Failed: ' + str(pip['tofail'].failed), level='critical')
    assert pip['tofail'].failed == True
    lg(str(get_pipeline()), level=0)


def test_output():
    """Print the outputs from the commands"""
    pip = get_pipeline()
    assert pip['write_something'].out == 'call1\n'
    lg(str(repr(pip['tofail'])), level=2)
    assert pip['tofail'].err.endswith('No such file or directory')
    for step in pip[:4]:
        assert step.out is not None


def test_simple_command():
    """Should be able to add a piped command as a single string."""
    pip = get_pipeline()
    pip.add('for i in $(ls); do echo hi $i; done | sort')
    pip.run('for')
    assert pip['for'].out
    assert pip['for'].done == True


def test_display():
    """Print all string objects from classes"""
    pip = get_pipeline()
    lg(str(pip), level=1)
    for step in pip:
        lg(str(step), level=1)

def test_fail_add():
    """Add a nonexistent path, expect failure."""
    pip = get_pipeline()
    with pytest.raises(pl.PathError):
        pip.add_command('bob_dylan')
    with pytest.raises(pl.Pipeline.PipelineError):
        pip['jack'] = 42
    pip['john'] = ('ls', 'jkldsf')
    with pytest.raises(pl.Command.CommandFailed):
        pip.run('john')

def test_remove_files():
    """Remove the pickle file"""
    os.remove(PIPELINE_FILE)
    #  os.remove(PIPELINE_FILE + '.log')
