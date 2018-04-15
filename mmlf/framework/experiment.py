# Maja Machine Learning Framework
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

# Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
# Created: 2011/04/27

""" Classes and functions used for MMLF experiments. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import time
from Queue import Empty
from threading import Thread, currentThread, Lock
from multiprocessing import Process, Manager, cpu_count

import mmlf

################################################################################
# Workaround for interrupted syscalls based on 
# https://groups.google.com/group/comp.lang.python/browse_thread/thread/ae3cf5d475c4f1ba
################################################################################ 
import errno
from multiprocessing.queues import Queue
def retry_on_eintr(function, *args, **kw):
    while True:
        try:
            return function(*args, **kw)
        except IOError, e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise

class RetryQueue(Queue):
    """Queue which will retry if interrupted with EINTR."""
    def get(self, block=True, timeout=None):
        return retry_on_eintr(Queue.get, self, block, timeout)
################################################################################ 

class WorldRunner(object):
    """ Class which encapsulates running of a world in an MMLF experiment.
    
    This class contains code for supporting the distributed, concurrent execution
    of worlds in an MMLF experiment. One instance of the class WorldRunner
    should be created for every run of a world. By calling this instance,
    the world is executed. This may happen in a separate thread or process since
    the instances of WorldRunner can be passed between threads and processes.
    
    **Parameters** 
        :worldConfigObject: : The world configuration (in a dictionary)  
        :numberOfEpisodes: : The number of episodes that should be conducted in the world
        :exceptionOccurredSignal: : An optional PyQt signal to which exceptions can be send    
    """
    
    def __init__(self, worldConfigObject, numberOfEpisodes, 
                 exceptionOccurredSignal=None):
        self.worldConfigObject = worldConfigObject
        self.numberOfEpisodes = numberOfEpisodes
        self.exceptionOccurredSignal = exceptionOccurredSignal
        
    def __call__(self, manager, observableQueue, updateQueue, forkQueue,
                 worldNumber, runNumber, useGUI=True):
        print "", # This has to be here. Whyever???
        # Choose different random seed for each run
        import random
        import numpy.random
        random.seed(runNumber)
        numpy.random.seed(runNumber)
        # Inform parent process via updateQueue that we actually forked
        forkQueue.put((worldNumber, runNumber))
        
        # Give global agent and run numbers as attributes to
        # threads since they are later on required by experiment
        # viewer and observables
        ct = currentThread()
        ct.worldNumber = worldNumber
        ct.runNumber = runNumber
        
        # Add an observer for OBSERVABLES that adds an ObservableProxy
        # for each Observable and puts this ObservableProxy into
        # the queue to the main process
        from mmlf.framework.observables import OBSERVABLES, FloatStreamObservable
        from mmlf.gui.multiprocessing_observables_utils import ObservableProxy
        def registerToObservables(observable, action):
            if isinstance(observable, FloatStreamObservable) \
                    and action == 'added':
                observableProxy = ObservableProxy(observable, manager)
                observableQueue.put(observableProxy)
                
                time.sleep(0.1) # TODO: Why is this necessary?? Somehow related to queues and spawning/forking?
                observableProxy.setObserverQueue(updateQueue)
        
        OBSERVABLES.addObserver(registerToObservables)
        
        # Load world
        world = mmlf.loadWorld(worldConfigObject=self.worldConfigObject,
                               useGUI=useGUI, keepObservers=[registerToObservables])
        
        # Set log level
        mmlf.setupConsoleLogging(level="warning")
        
        # Run specified number of episodes in the world
        mmlf.log.debug("Running %s episodes in world..." % self.numberOfEpisodes)
        try:
            world.run(self.numberOfEpisodes)
        except Exception, e:
            errorMessage = e.__class__.__name__ + ": " + str(e)
            if self.exceptionOccurredSignal:
                self.exceptionOccurredSignal.emit(errorMessage)
            raise
             
        mmlf.log.debug("Running world...Done!")  
        
        
def runExperiment(experimentConfig, observableQueue, updateQueue, 
                  exceptionOccurredSignal, useGUI=True):
    """ Handles the execution of an experiment.
    
    This function handles the concurrent or sequential execution of an experiment.
    
    **Parameters** 
        :experimentConfig: : The experiment configuration (in a dictionary)  
        :observableQueue: : A multiprocessing.Queue. Used for informing the main 
                            process (e.g. the GUI) of observables created in the
                            world runs.                            
        :updateQueue: : A multiprocessing.Queue. Used for informing the main 
                        process (e.g. the GUI) of changes in observables.
        :exceptionOccurredSignal: : An optional PyQt signal to which exceptions 
                                    can be send    
    """ 
    # inf must be imported since text that is entered by user in
    # GUI might contain string "inf"
    from numpy import inf
    numberOfRuns = eval(experimentConfig["runsPerWorld"])
    numberOfEpisodesPerRun = eval(experimentConfig["episodesPerRun"])
    executionOrder = experimentConfig["concurrency"]
    numParallelProcesses = min(eval(experimentConfig["parallelProcesses"]),
                               cpu_count())
    
    # A manager that provides a way to create data which can be shared 
    # between different processes
    manager = Manager()
    
    # The set of all running processes started for this experiment
    processes = set()
    processesLock = Lock()

    def launchConcurrentProcess(worldRunner, observableQueue, updateQueue, 
                                worldNumber, runNumber, useGUI):
        """ Run world in a separate process."""
        
        # Add a dummy to processes set to signal that we want to start   
        processesLock.acquire()
        processes.add((worldNumber, runNumber))
        processesLock.release()
                                     
        # Start subprocess and see if we have actually forked
        # NOTE: For some weird reason, the subprocess is sometimes not 
        #       forked correctly. We try forking until it works.
        notForked = True
        forkQueue = RetryQueue() 
        while notForked:
            process = Process(target=worldRunner, 
                              args=(manager, observableQueue, updateQueue,
                                    forkQueue, worldNumber, runNumber, useGUI)) 
            process.start()
            
            try:
                assert (forkQueue.get(timeout=3) == (worldNumber, runNumber))
            except Empty:
                mmlf.log.warning("Subprocess for run %s of world "
                                 "%s did not fork. We try again..." 
                                     % (runNumber, worldNumber))
                process.terminate()
                continue
                
            notForked = False 
            
        # Add process to list of running processes
        processesLock.acquire()
        processes.remove((worldNumber, runNumber))
        processes.add(process)
        processesLock.release()        
        # Wait for termination of process            
        process.join()
        
        # Remove process from list of running processes
        processesLock.acquire()
        processes.remove(process)
        processesLock.release()
        
    def launchSequentialProcess(worldRunner, observableQueue, updateQueue, 
                                worldNumber, runNumber, useGUI):                
        """ Run world in the same process since processes run sequentially. """
        forkQueue = Queue() 
        worldRunner(manager, observableQueue, updateQueue, forkQueue,
                    worldNumber, runNumber, useGUI)
        # Empty fork queue
        try:
            while True:
                forkQueue.get(False)
        except:
            pass
    
    # Run all worlds
    runCounter = 0
    totalRuns = numberOfRuns * len(experimentConfig["worlds"].keys())
    for worldName, worldConfigObject in experimentConfig["worlds"].iteritems():
        # World runner for the given setting
        worldRunner = WorldRunner(worldConfigObject, 
                                  numberOfEpisodesPerRun,
                                  exceptionOccurredSignal)
        # Perform numberOfRuns runs for agent setting 1
        for runNumber in range(numberOfRuns):
            # Allow only numParallelProcesses parallel running processes
            while True:
                mmlf.log.info("Total runs: %s\tFinished runs: %s\tActive runs: %s" \
                    % (totalRuns, runCounter - len(processes), len(processes)))
                if len(processes) < numParallelProcesses:
                    break
                time.sleep(1.0) # Avoid busy loop
                
            runCounter += 1
            # Run world launcher in a separate thread that waits for this
            # world to terminate
            if executionOrder == "Concurrent":
                thread = Thread(target=launchConcurrentProcess, 
                                args=(worldRunner, observableQueue, 
                                      updateQueue,worldName, runNumber, useGUI))
                thread.start()
                time.sleep(1.0) # Processes must not be started too quickly
            else:
                thread = Thread(target=launchSequentialProcess, 
                                args=(worldRunner, observableQueue, 
                                      updateQueue,worldName, runNumber, useGUI))
                thread.start()
                thread.join()
    
    while len(processes) > 0:
        mmlf.log.info("Total runs: %s\tFinished runs: %s\tActive runs: %s" \
                    % (totalRuns, runCounter - len(processes), len(processes)))
        time.sleep(1.0) # Avoid busy loop
        
    mmlf.log.info("Total runs: %s\tFinished runs: %s\tActive runs: %s" \
                    % (totalRuns, runCounter - len(processes), len(processes)))
    mmlf.log.info("Experiment finished!")
