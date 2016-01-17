import os
import sys
import pipeline as pl
import pytest
from logme import log

PIPELINE_FILE = 'test.test'
LOGFILE = 'test_pipeline.log'

###############################################################################
#                           Non-Test Functions                                #
###############################################################################

def lg(msg):
    """Run logme log with LOGFILE"""
    log(msg, LOGFILE)


def write_something(msg):
    """Write msg to stdout"""
    return(msg + '\n')


def get_pipeline():
    """Get the pipeline object."""
    return  pl.get_pipeline(PIPELINE_FILE)


def create_pipeline():
    """Create pipeline"""
    pl.get_pipeline(PIPELINE_FILE)


def add_to_pipeline(pip):
    """Add some steps to the pipeline"""
    pip.add('ls')
    pip.add('ls', '~', name='bob')
    pip.add_function(write_something, 'call1')
    pip.add(write_something, args=('call2',), name='run2')
    assert len(pip) == 4
    pip.save()


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
    add_to_pipeline(get_pipeline())


def test_restore():
    """Make sure methods still exist"""
    pip = get_pipeline()
    assert len(pip) == 4
    assert pip['bob'].command == pl.get_path('ls')
    assert pip['bob'].args == '~'
    assert pip['bob'].done == False
    assert isinstance(pip['bob'], pl.Command)
    assert isinstance(pip['run2'], pl.Function)
    assert str(pip['write_something'].command).startswith(
        '<function write_something')
    assert pip['run2'].args == ('call2',)


def test_run():
    """Execute pipeline steps."""
    get_pipeline().run_all()
    lg(str(get_pipeline()))


def test_output():
    """Print the outputs from the commands"""
    pip = get_pipeline()
    assert pip['write_something'].out == 'call1\n'
    for step in pip:
        assert step.out is not None

def test_display():
    """Print all string objects from classes"""
    pip = get_pipeline()
    lg(str(pip))
    for step in pip:
        lg(str(step))

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
