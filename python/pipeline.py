"""
Easily manage a complex pipeline with python.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 1.0.1
       CREATED: 2016-14-15 16:01
 Last modified: 2016-01-18 00:30

   DESCRIPTION: Classes and functions to make running a pipeline easy

 USAGE EXAMPLE: import pipeline as pl
                pipeline = get_pipeline(file)  # file holds a pickled pipeline
                pipeline.add('bed_to_vcf', ('bed_file', 'vcf_file'))
                pipeline.add('cat bed_file | bed_to_vcf > vcf_file',
                             name='bed2vcf')
                def my_test():
                    return True if os.path.isfile('vcf_file') else False
                pipeline.add('cat bed_file | bed_to_vcf > vcf_file',
                             name='bed2vcf2', failtest=my_test)
                def my_fun(no1, no2):
                    return no1 + no2
                pipeline.add(my_fun, (1, 2))
                pipeline.add(print, 'hi',     # Only run print('hi') if
                             pretest=my_test) # my_test returns True
                pipeline.run_all()
                pipeline['my_fun'].out  # Will return 3
                print(pipeline)  # Will print detailed pipeline stats

============================================================================
"""
import os
import time
import sys
from datetime import datetime as dt
from subprocess import call
from subprocess import Popen
from subprocess import PIPE
try:
    import cPickle as pickle
except ImportError:
    import pickle
from logme import log as lm  # In the MikeDacre/mike_tools/python repo
from logme import LoggingException

############################
#  Customizable constants  #
############################

DEFAULT_FILE = './pipeline_state.pickle'
DEFAULT_PROT = 2  # Support python2 pickling, can be 4 if using python3 only
LOG_LEVEL    = 'info'  # Controls level of logging


###############################################################################
#                               Pipeline Class                                #
###############################################################################


class Pipeline(object):

    """A class to store and save the state of the current pipeline.

    Do not call directly, instead access using the get_pipeline() function.
    """

    def __init__(self, pickle_file=DEFAULT_FILE, root='.', prot=DEFAULT_PROT):
        """Setup initial variables and save."""
        self.step     = 'start'
        self.steps    = {}  # Command object by name
        self.order    = ()  # The order of the steps
        self.current  = None  # The current pipeline step
        self.file     = pickle_file
        self.logfile  = pickle_file + '.log'  # Not set by init
        self.loglev   = LOG_LEVEL
        self.root_dir = os.path.abspath(str(root))
        self.prot     = int(prot)  # Can change version if required
        self.save()

    #####################
    #  Step Management  #
    #####################

    def save(self):
        """Save state to the provided pickle file.

        This will save all of the Step classes also, and should
        be called on every modification
        """
        with open(self.file, 'wb') as fout:
            pickle.dump(self, fout, protocol=self.prot)

    def add(self, command, args=None, name=None, kind='', store=True,
            failtest=None, pretest=None):
        """Wrapper for add_command and add_function.

        Attempts to detect kind, defaults to function

        :command:  A shell script, command, or callable function.
        :args:     Args to pass to command or function, cannot be used if
                   command is a shell script.
        :name:     An optional name for the step, defaults to command.
        :kind:     Force either 'function' or 'command'
        :store:    Store all output in '.out'. This is always true for
                   functions.
        :failtest: An optional function to test for success. Will always be run
                   right after successful execution of this step. This can be
                   either a single function call or a tuple of (function_call,
                   args). Function call must be an actual function, not a
                   string or anything else.
                   If failtest function returns True or 0, the test is a
                   success, otherwise it is assumed that this step failed.
        :pretest:  Like failtest but run before step, must be True for step to
                   run.
        """
        if not kind:
            if isinstance(command, str):
                kind = 'command'
            else:
                kind = 'function'
        if kind == 'command':
            self.add_command(command, args, name, store, failtest, pretest)
        elif kind == 'function':
            self.add_function(command, args, name, store, failtest, pretest)
        else:
            raise self.PipelineError('Invalid step type: {}'.format(kind),
                                     self.logfile)

    def delete(self, name):
        """Delete a step by name."""
        if name in self.steps:
            self.steps.pop(name)
        else:
            self.log('{} not in steps dict'.format(name), level='warn')
        if name in self.order:
            ind = self.order.index(name)
            self.order = self.order[:ind] + self.order[ind + 1:]
        else:
            self.log('{} not in order tuple'.format(name), level='warn')
        self.save()

    def add_command(self, program, args=None, name=None, store=True,
                    failtest=None, pretest=None):
        """Add a simple pipeline step via a Command object."""
        name = name if name else program.split(' ')[0].split('/')[-1]
        if name not in self.steps:
            self.steps[name] = Command(program, args, store, parent=self,
                                       failtest=failtest, pretest=pretest,
                                       name=name)
            self.order = self.order + (name,)
        else:
            self.log(('{} already in steps. Please choose another ' +
                      'or delete it').format(name), level='error')
        self._get_current()
        self.save()

    def add_function(self, function_call, args=None, name=None, store=True,
                     failtest=None, pretest=None):
        """Add a simple pipeline step via a Command object."""
        if not name:
            parts = str(function_call).strip('<>').split(' ')
            parts.remove('function')
            try:
                parts.remove('built-in')
            except ValueError:
                pass
            name = parts[0]
        if name not in self.steps:
            self.steps[name] = Function(function_call, args, store,
                                        parent=self, failtest=failtest,
                                        pretest=pretest, name=name)
            self.order = self.order + (name,)
        else:
            self.log(('{} already in steps. Please choose another ' +
                      'or delete it').format(name), level='error')
        self._get_current()
        self.save()

    #############
    #  Running  #
    #############

    def run_all(self, force=False):
        """Run all steps in order.

        Skip already completed steps unless force is True, in which case
        run all anyway
        """
        self._get_current()
        self.save()
        for step in self.order:
            if not force and self.steps[step].done:
                continue
            self.run(step)
        self._get_current()
        self.save()

    def run(self, step='current'):
        """Run a specific step by name.

        If 'current' run the most recent 'Not run' or 'failed' step.
        """
        self._get_current()
        if step == 'current':
            cur = None
            if not self.order:
                self.log('No steps added yet, not running', level='warn')
                return
            for step in self:
                if not step.done:
                    cur = step.name
                    break
            if not cur:
                self.log('All steps already complete, not running',
                         level='warn')
                return
            self.steps[cur].run()
        elif step in self.order:
            try:
                self.steps[step].run()
            except:
                self.log('Step {} failed!'.format(step), 'critical')
                self.save()
                raise
        else:
            raise self.PipelineError('{} Is not a valid pipeline step'.format(
                step), self.logfile)
        self._get_current()
        self.save()

    ###############
    #  Internals  #
    ###############

    def log(self, message, level='debug'):
        """Wrapper for logme log function."""
        lm(message, logfile=self.logfile, level=level, min_level=self.loglev)

    def _get_current(self):
        """Set self.current to most recent 'Not run' or 'Failed' step."""
        if self.order:
            for step in self:
                if not step.done or step.failed:
                    self.current = step.name
                    return
        self.current = None

    def __getitem__(self, item):
        """Return a Step from self.steps."""
        if item in self.order:
            return self.steps[item]
        else:
            return None

    def __setitem__(self, name, args):
        """Call self.add() indirectly.

        To use, args must be either a string containing a command,
        or a tuple/dict compatible with add(). It is better to call add()
        directly.
        """
        if isinstance(args, str):
            self.add(args, name=name)
        elif isinstance(args, (tuple, list)):
            self.add(*args, name=name)
        elif isinstance(args, dict):
            self.add(name=name, **args)
        else:
            raise self.PipelineError('args must be a tuple, dict, or str',
                                     self.logfile)
        self.save()

    def __delitem__(self, item):
        """Call self.delete indirectly."""
        self.delete(item)

    def __contains__(self, item):
        """Check in self.order."""
        return True if item in self.order else False

    def __iter__(self):
        """Iterate through steps."""
        for step in self.order:
            yield self.steps[step]

    def __len__(self):
        """Return number of steps."""
        return len(self.order)

    def __str__(self):
        """Simple information about the class."""
        output = 'Pipeline:\n'
        if self.steps:
            names = ()
            steps = ()
            statuses = ()
            for step in self.order:
                names = names + (step,)
                steps = steps + (self.order.index(step),)
                stat = 'Done' if self.steps[step].done else 'Not run'
                stat = 'FAILED' if self.steps[step].failed else stat
                statuses = statuses + (stat,)
            len1 = 7
            len2 = max(len(i) for i in names) + 4
            output = output + ('Step'.ljust(len1) + 'Name'.ljust(len2) +
                               'Status\n')
            for step in steps:
                output = output + (str(step).ljust(len1) +
                                   names[step].ljust(len2) +
                                   statuses[step] + '\n')
        else:
            output = output + "No steps assigned"
        return output

    def __repr__(self):
        """Detailed information about the class."""
        output = ("<Pipeline(file={}, steps={}, " +
                  "done={}, failed={})>").format(
                      self.file,
                      len(self.order),
                      len([i for i in self.steps.values() if i.done]),
                      len([i for i in self.steps.values() if i.failed]))
        return output

    class PipelineError(LoggingException):

        """Failed pipeline steps."""

        pass


###############################################################################
#                    Classes for Individual Pipeline Steps                    #
###############################################################################


class Step(object):

    """A single pipeline step.

    Not intended to be used directly, generally Function or Command classes
    should be used instead as they inherit from here.If you use this class
    directly you must add a run method.
    """

    def __init__(self, command, args=None, store=True, parent=None,
                 failtest=None, pretest=None, name='unknown step'):
        """Set the program path and arguments.

        :command:  The command, script, or function call to be executed.
        :args:     Optional arguments to pass with command.
        :store:    Capture output to out and (if shell executed) err.
        :parent:   The Pipeline object that owns this child.
        :failtest: An optional function call (or tuple of (function, args))
                   to be executed following run. Must return True on success
                   or False on failure.
        :pretest:  Like pretest but run before execution, must be True to run.
        """
        self.command     = command
        self.args        = args
        self.store       = store   # Store output on run
        self.parent      = parent  # The Pipeline object that created us
        self.name        = name    # Should match name in parent dictionary
        self.done        = False   # We haven't run yet
        self.failed      = False
        self.failed_pre  = False
        self.failed_test = False
        self.start_time  = None
        self.end_time    = None
        self.code        = None
        self.out         = None    # STDOUT or returned data
        self.err         = None    # STDERR only
        self.logfile     = self.parent.logfile if self.parent.logfile else None
        self.loglev      = self.parent.loglev if self.parent.loglev \
            else LOG_LEVEL
        # Test the tests now to avoid frustration
        if failtest:
            self.failtest = failtest
            self._test_test(self.failtest)
        else:
            self.failtest = None
        if pretest:
            self.log('Pretest added', level=0)
            self.pretest = pretest
            self._test_test(self.pretest)
        else:
            self.pretest = None

    #####################
    #  Execution Tests  #
    #####################

    def run_test(self, test, raise_error_on_fail=True):
        """Run a test function.

        Will evalucate to success if test function returns True or 0, failure
        on any other return. Any exceptions raised during the handling will
        cause failure.

        If raise_error_on_fail is True, a FailedTest Exception or the
        function's own Exception will be raised. Otherwise they will not.
        """
        self.log('Running test ' + str(test), level=0)
        self._test_test(test)
        # Run the function
        out = False
        if isinstance(test, tuple):
            out = run_function(*test)
        else:
            out = run_function(test)

        # Test the output
        if out is True or out is 0:
            return True
        else:
            if self.parent:
                self.parent.save()
            if raise_error_on_fail:
                raise self.FailedTest(
                    'Fail test failed with function output {}'.format(out),
                    self.logfile)
            else:
                return False

    def run_fail_test(self, raise_error_on_fail=True):
        """Run a fail test with run_test and set self.failed & self done."""
        if not self.failtest:
            raise self.StepError('Cannot run failtest if failtest function ' +
                                 'not assigned', logfile=self.logfile)
        try:
            out = self.run_test(self.failtest, raise_error_on_fail)
        except:
            self.done        = False
            self.failed      = True
            self.failed_test = True
            if self.parent:
                self.parent.save()
            if raise_error_on_fail:
                raise
            else:
                return False

        if out is True:
            self.done   = True
            self.failed = False
            if self.parent:
                self.parent.save()
            return True
        else:
            self.done        = False
            self.failed      = True
            self.failed_test = True
            if self.parent:
                self.parent.save()
            return False

    def run_pre_test(self, raise_error_on_fail=True):
        """Run a fail test with run_test and set self.failed & self done."""
        if not self.pretest:
            raise self.StepError('Cannot run pretest if pretest function ' +
                                 'not assigned', logfile=self.logfile)
        try:
            out = self.run_test(self.pretest, raise_error_on_fail)
        except:
            self.failed_pre = True
            if self.parent:
                self.parent.save()
            if raise_error_on_fail:
                raise
            else:
                return False

        if out is True:
            self.failed_pre = False
            if self.parent:
                self.parent.save()
            return True
        else:
            self.failed_pre = True
            if self.parent:
                self.parent.save()
            return False

    def log(self, message, level='debug'):
        """Wrapper for logme log function."""
        args = {'level': level, 'min_level': self.loglev}
        if self.logfile:
            args.update({'logfile': self.logfile})
        message = self.name + ' > ' + str(message)
        lm(message, **args)

    ###############
    #  Internals  #
    ###############

    def _test_test(self, test):
        """Test a single test instance to make sure it is usable."""
        if isinstance(test, tuple):
            if len(test) != 2:
                raise self.StepError('Test must have only two values:' +
                                     'the function call, and the args.\n' +
                                     "It's current value is:" +
                                     "\n{}".format(test),
                                     self.logfile)
            function_call = test[0]
        else:
            function_call = test
        if not hasattr(function_call, '__call__'):
            raise self.StepError(('Function must be callable, but {} ' +
                                  'is of type {}').format(
                                      function_call,
                                      type(function_call)), self.logfile)

    def __str__(self):
        """Display simple class info."""
        runmsg = 'Complete' if self.done else 'Not run'
        runmsg = 'Failed' if self.failed else runmsg
        output = "{}, args: {} -- State: {}\n".format(self.command,
                                                      self.args,
                                                      runmsg.upper())
        if self.pretest:
            if self.failed_pre:
                fmessage = '[FAILED]'
            elif self.done:
                fmessage = '[DONE]'
            else:
                fmessage = ''
            output = output + 'Pretest: {} {}\n'.format(self.pretest,
                                                        fmessage)
        if self.failtest:
            if self.failed_test:
                fmessage = '[FAILED]'
            elif self.done:
                fmessage = '[DONE]'
            else:
                fmessage = ''
            output = output + 'Failtest {} {}\n'.format(self.failtest,
                                                        fmessage)

        if self.done or self.failed:
            timediff = str(dt.fromtimestamp(self.end_time) -
                           dt.fromtimestamp(self.start_time))
            output = output + "\n\n{0:<11}{1}\n".format(
                'Ran on:',
                time.ctime(self.start_time))
            if self.code is not None:
                output = output + "{0:<11}{1}\n".format('Exit code:',
                                                        self.code)
            output = output + "{0:<11}{1}".format('Runtime:', timediff)
            if self.failed:
                output = output + "\n\n\tWARNING --> Command Failed!"
        return output

    def __repr__(self):
        """Print output if already run, else just args."""
        if self.done:
            stat = 'Done'
        elif self.failed:
            stat = 'Failed'
        else:
            stat = 'Not run'
        pretest = str(self.pretest) + ' [FAILED]' if self.failed_pre \
            else str(self.pretest)
        failtest = str(self.failtest) + ' [FAILED]' if self.failed_test \
            else str(self.failtest)
        return ("<Step(Class={0}, Command={1}, Args='{2}', Pretest={3}, " +
                "Failtest={4}, Run={5}, Code={6}, Output={7}, STDERR={8}, " +
                "Store={9})>").format(
                    type(self), self.command, self.args, pretest,
                    failtest, stat, self.code,
                    True if self.out else False, True if self.err else False,
                    self.store)

    ################
    #  Exceptions  #
    ################

    class FailedTest(LoggingException):

        """Failed during failtest."""

        pass

    class StepError(LoggingException):

        """Failed to build the command."""

        pass

####################################################################
#  Types of Step, these should be called directly instead of Step  #
####################################################################


class Function(Step):

    """A single function as a pipeline step.

    NOTE: The command argument must be an actual function handle,
          not a string
    """

    def __init__(self, function, args=None, store=True, parent=None,
                 failtest=None, pretest=None, name='unknown function'):
        """Build the function."""
        # Make sure function is callable
        if not hasattr(function, '__call__'):
            raise self.StepError(('Function must be callable, but {} ' +
                                  'is of type {}').format(function,
                                                          type(function)),
                                 self.logfile)
        # Make sure args are a tuple
        if args:
            if not isinstance(args, tuple):
                args = (args,)
        super(Function, self).__init__(function, args, store, parent, failtest,
                                       pretest, name)

    def run(self, kind=''):
        """Execute the function with the provided args.

        Types:
            check - Just run, if function fails, traceback will occur
                    output is still stored in self.out
            get   - return output
        """
        # Run pretest first if available
        if self.pretest:
            if not self.run_pre_test():  # Will throw exception on failure
                return                   # Definitely abort on fail

        # Set kind from storage option
        if not kind:
            kind = 'get' if self.store else 'check'

        # Run the function
        self.start_time = time.time()
        try:
            if self.args:
                self.out = run_function(self.command, self.args)
            else:
                self.out = run_function(self.command)
        except:
            self.failed = True
            raise
        else:
            self.done = True
        finally:
            self.end_time = time.time()
            if self.parent:
                self.parent.save()

        # Run the failtest if available
        if self.failtest:
            self.run_fail_test()
        if kind == 'get':
            return self.out


class Command(Step):

    """A single external command as a pipeline step."""

    def __init__(self, command, args=None, store=True, parent=None,
                 failtest=None, pretest=None, name='unknown command'):
        """Build the command."""
        logfile = parent.logfile if parent else sys.stderr
        # Make sure command exists if not a shell script
        if len(command.split(' ')) == 1:
            command = get_path(command, logfile)
        elif args:
            raise self.StepError('Cannot have a multi-word command ' +
                                 'and an argument string for a command.\n' +
                                 'Pick one of the other.',
                                 logfile)

        # Make sure args can be used
        if args:
            if not isinstance(args, (tuple, list, str)):
                raise self.StepError('args must be string, list, or tuple' +
                                     ' but is {}'.format(type(args)),
                                     logfile)

        # Initialize the whole object
        super(Command, self).__init__(command, args, store, parent, failtest,
                                      pretest, name)

    def run(self, kind=''):
        """Run the command.

        Shell is always True, meaning redirection and shell commands will
        function as expected.
        Types:
            check - check_call output not saved
            get   - return output
        """
        # Run pretest first if available
        if self.pretest:
            if not self.run_pre_test():  # Will throw exception on failure
                return                   # Definitely abort on fail

        # Set kind from storage option
        if not kind:
            kind = 'get' if self.store else 'check'

        # Make a string from the command as we run with shell=True
        if self.args:
            if isinstance(self.args, (list, tuple)):
                args = ' '.join(self.args)
            elif isinstance(self.args, str):
                args = self.args
            else:
                raise self.StepError('Invalid argument type',
                                     self.parent.logfile)
            command = self.command + ' ' + args
        else:
            command = self.command

        # Actually run the command
        self.start_time = time.time()
        try:
            if kind == 'get':
                self.code, self.out, self.err = run_cmd(command)
            elif kind == 'check':
                self.code = call(command, shell=True)
        except:
            self.failed = True
            raise
        finally:
            self.end_time = time.time()
            if self.parent:
                self.parent.save()

        if self.code == 0:
            self.done = True
        else:
            self.failed = True
            err = '{} Failed.\nRan as:\n{}'.format(self.command,
                                                   command)
            if self.out:
                err = err + '\nOutput:\n{}'.format(self.out)
            if self.err:
                err = err + '\nSTDERR:\n{}'.format(self.err)
            if self.parent:
                self.parent.save()
            raise self.CommandFailed(err, self.parent.logfile)

        # Run the failtest if available
        if self.failtest:
            self.run_fail_test()
        if kind == 'get':
            return self.out

        return self.code

    ################
    #  Exceptions  #
    ################

    class CommandFailed(LoggingException):

        """Executed command returned non-zero."""

        pass


###############################################################################
#                         Data Management Functions                           #
###############################################################################


def restore_pipeline(pickle_file=DEFAULT_FILE):
    """Return an AlleleSeqPipeline object restored from the pickle_file.

    prot can be used to change the default protocol
    """
    with open(pickle_file, 'rb') as fin:
        return pickle.load(fin)


def get_pipeline(pickle_file=DEFAULT_FILE, root='.', prot=DEFAULT_PROT):
    """Create or restore a pipeline at pickle_file.

    If pickle file exists, restore it, else make a new session
    and save it. Return AlleleSeqPipeline object
    """
    if os.path.isfile(pickle_file):
        return restore_pipeline(pickle_file)
    else:
        pipeline = Pipeline(pickle_file=os.path.abspath(str(pickle_file)),
                            root=os.path.abspath(str(root)),
                            prot=int(prot))
        pipeline.save()
        return pipeline

###############################################################################
#                               Other Functions                               #
###############################################################################


def run_cmd(cmd):
    """Run command and return status, output, stderr.

    cmd is run with shell, so must be a string.
    """
    pp = Popen(str(cmd), shell=True, universal_newlines=True,
               stdout=PIPE, stderr=PIPE)
    out, err = pp.communicate()
    code = pp.returncode
    if out[-1:] == '\n':
        out = out[:-1]
    if err[-1:] == '\n':
        err = err[:-1]
    return code, out, err


def run_function(function_call, args=None):
    """Run a function with args and return output."""
    if not hasattr(function_call, '__call__'):
        raise FunctionError('{} is not a callable function.'.format(
            function_call))
    if args:
        if isinstance(args, (tuple, list)):
            out = function_call(*args)
        elif isinstance(args, dict):
            out = function_call(**args)
        else:
            out = function_call(args)
    else:
        out = function_call()
    return out


def get_path(executable, log=None):
    """Use `which` to get the path of an executable.

    Raises PathError on failure
    :returns: Full absolute path on success
    """
    code, out, err = run_cmd('which {}'.format(executable))
    if code != 0 or err == '{} not found'.format(executable):
        raise PathError('{} is not in your path'.format(executable), log)
    else:
        return os.path.abspath(out)


###############################################################################
#                           Other Exceptions                                  #
###############################################################################


class PathError(LoggingException):

    """Command not in path."""

    pass


class FunctionError(LoggingException):

    """Function call failed."""

    pass
