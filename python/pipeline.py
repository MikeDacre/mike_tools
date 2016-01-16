"""
============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 1.0
       CREATED: 2016-14-15 16:01
 Last modified: 2016-01-15 18:00

   DESCRIPTION: Classes and functions to make running a pipeline easy

         USAGE: import pipeline as pl
                pipeline = get_pipeline(file)  # file holds a pickled pipeline
                pipeline.add_step('bed_to_vcf', command, args)
                pipeline.steps['bed_to_vcf'].run()

============================================================================
"""
import os
import sys
import time
from datetime import datetime as dt
from subprocess import getoutput
from subprocess import getstatusoutput as chk
try:
    import cPickle as pickle
except ImportError:
    import pickle

DEFAULT_FILE = './pipeline_state.pickle'
DEFAULT_PROT = 2  # Support python2 pickling


###############################################################################
#                               Pipeline Class                                #
###############################################################################


class Pipeline(object):
    """ A class to store and save the state of the current pipeline """
    def __init__(self, pickle_file=DEFAULT_FILE, root='.'):
        self.step     = 'start'
        self.steps    = {}  # Command object by name
        self.order    = ()  # The order of the steps
        self.file     = pickle_file
        self.root_dir = os.path.abspath(root)
        self.prot     = DEFAULT_PROT  # Can change version if required

    #####################
    #  Step Management  #
    #####################

    def save(self):
        """ Save state to the provided pickle file """
        with open(self.file, 'wb') as fout:
            pickle.dump(self, fout, protocol=self.prot)

    def add_step(self, command, args=None, name=None, kind='function'):
        """ Wrapper for add_command and add_function
            Defaults to function """
        if kind == 'command':
            self.add_command(command, args, name)
        elif kind == 'function':
            self.add_function(command, args, name)
        else:
            raise Exception('Invalid step type: {}'.format(kind))

    def del_step(self, name):
        """ Delete a step by name """
        if name in self.steps:
            self.steps.pop(name)
        else:
            sys.stderr.write('{} not in steps dict\n'.format(name))
        if name in self.order:
            ind = self.order.index(name)
            self.order = self.order[:ind] + self.order[ind + 1:]
        else:
            sys.stderr.write('{} not in order tuple\n'.format(name))

    def add_command(self, program, args=None, name=None):
        """ Add a simple pipeline step via a Command object """
        name = name if name else program.split('/')[-1]
        if name not in self.steps:
            self.steps[name] = Command(program, args)
            self.order = self.order + (name,)
        else:
            sys.stderr.write(('{} already in steps. Please choose another ' +
                              'or delete it\n').format(name))

    def add_function(self, function_call, args=None, name=None):
        """ Add a simple pipeline step via a Command object """
        name = name if name else str(function_call).lstrip('<').split(' ')[1]
        if name not in self.steps:
            self.steps[name] = Function(function_call, args)
            self.order = self.order + (name,)
        else:
            sys.stderr.write(('{} already in steps. Please choose another ' +
                              'or delete it\n').format(name))

    #############
    #  Running  #
    #############

    def run_all(self, force=False):
        """ Run all steps in order, skip already completed steps unless
            force is True, in which case run all anyway
        """
        pass

    def __repr__(self):
        names = ()
        steps = ()
        statuses = ()
        for name, step in self.steps:
            names = names + (name,)
            steps = steps + (self.order.index(name),)
            statuses = statuses + ('Done' if step.done else 'Not run')
        len1 = max(len(i) for i in names) + 2
        len2 = max(len(i) for i in steps) + 2
        len3 = max(len(i) for i in statuses) + 2
        output = ('Step'.ljust(len1) + 'Name'.ljust(len2) +
                  'Status'.ljust(len3) + '\n')
        for step in steps:
            output = (str(step).ljust(len1) + names[step].ljust(len2) +
                      statuses[step].ljust(len3) + '\n')
        return output


class Step(object):
    """ A single pipeline step.
        Generally Function or Command classes should be used instead
    """

    def __init__(self, command, args=None):
        """ Set the program path and arguments """
        self.done       = False  # We haven't run yet
        self.command    = command
        self.args       = args
        self.kind       = None
        self.start_time = None
        self.end_time   = None
        self.code       = None
        self.out        = None

    def __repr__(self):
        """ Print output if already run, else just args """
        output = "{0:<11}{1}\n".format('Program:', self.command)
        if self.args:
            output = output + "{0:<11}{1}\n".format('Args:', self.args)
        else:
            output = output + "{0:<11}{1}\n".format('Args:', 'None')
        if self.done:
            timediff = str(dt.fromtimestamp(self.end_time) -
                           dt.fromtimestamp(self.start_time))
            output = output + "{0:<11}{1}\n".format(
                'Ran on:',
                time.ctime(self.start_time))
            if self.code is not None:
                output = output + "{0:<11}{1}\n".format('Exit code:',
                                                        self.code)
            output = output + "{0:<11}{1}".format('Runtime:', timediff)
        else:
            output = output + "\nExecution has not begun yet"
        return output


class Function(Step):
    """ A single function as a pipeline step
        NOTE: The command argument must be an actual function handle,
              not a string
    """

    def __init__(self, command, args=None):
        super(Function, self).__init__(command, args)
        # Make sure args are a tuple
        if self.args:
            if not isinstance(self.args, tuple):
                self.args = (self.args,)
        self.kind = 'function'

    def run(self, kind='check'):
        """ Execute the function with the provided args
            Types:
                check - Just run, if function fails, traceback will occur
                get   - return output
        """
        self.start_time = time.time()
        if self.args:
            self.out = self.command(*self.args)
        else:
            self.out = self.command()
        self.code = None
        self.end_time = time.time()
        self.done = True
        if kind == 'get':
            return self.out


class Command(Step):
    """ A single external command as a pipeline step """

    def __init__(self, command, args=None):
        super(Command, self).__init__(command, args)

        # Make sure command exists
        self.command = getoutput('which {}'.format(command))
        if self.command == '{} not found'.format(command):
            raise Exception('{} is not in your path'.format(command))
        self.command = os.path.abspath(self.command)

        # Make sure args can be used
        if isinstance(self.args, (tuple, list)):
            self.args = ' '.join(self.args)
        if not isinstance(args, str):
            raise Exception('args must be string, list, or tuple')

        # Set type
        self.kind = 'command'

    def run(self, kind='check'):
        """ Run the command.
            Types:
                check - check_call
                get   - return output
        """
        if kind == 'get':
            command = self.command + self.args if self.args else self.command
            self.start_time = time.time()
            self.code, self.out = chk(command)
            self.end_time = time.time()
            self.done = True
            if self.code != 0:
                err = '{} Failed with args:\n{}'.format(self.command,
                                                        self.args)
                if self.out:
                    err = err + '\nOutput:\n{}'.format(self.out)
                raise self.CommandFailed(err)
            return self.out

    class CommandFailed(Exception):
        """ Failure Exception """
        pass


###############################################################################
#                               Data Management                               #
###############################################################################


def restore_pipeline(pickle_file=DEFAULT_FILE, prot=DEFAULT_PROT):
    """ Return an AlleleSeqPipeline object restored from the pickle_file
        prot can be used to change the default protocol """
    with open(pickle_file, 'rb') as fin:
        return pickle.load(fin, protocol=prot)


def get_pipeline(pickle_file=DEFAULT_FILE, root='.', prot=DEFAULT_PROT):
    """ If pickle file exists, restore it, else make a new session
        and save it. Return AlleleSeqPipeline object """
    if os.path.isfile(pickle_file):
        return restore_pipeline(pickle_file, prot)
    else:
        pipeline = Pipeline(pickle_file, os.path.abspath(root))
        pipeline.save()
        return pipeline
