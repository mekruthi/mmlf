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

""" Maja Machine Learning Framework

The Maja Machine Learning Framework (MMLF) is a general framework for 
problems in the domain of Reinforcement Learning (RL). It provides a 
set of RL related algorithms and a set of benchmark domains. 
Furthermore it is easily extensible and allows to automate 
benchmarking of different agents. 

Among the RL algorithms are TD(lambda), DYNA-TD, CMA-ES,
Fitted R-Max, and Monte-Carlo learning. MMLF contains different 
variants of the maze-world and pole-balancing problem class as 
well as the mountain-car testbed.

Further documentation is available under http://mmlf.sourceforge.net/

Contact the mailing list MMLF-support@lists.sourceforge.net
if you have any questions.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import sys
import os
import shutil
import time
import warnings
import yaml
import glob

import framework.mmlf_logging
log = framework.mmlf_logging.getLogger('FrameworkLog')
framework.mmlf_logging.setupConsoleLogging()

# Check for PyQt4 availability
try:
    from PyQt4 import QtGui, QtCore, QtWebKit
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
except ImportError:
    QtGui = None
    QtCore = None
    QtWebKit = None
    FigureCanvas = None    
    NavigationToolbar = None    
    
class RWAreaInitializationFailedException(Exception): 
    """ Exception raised if initialization of MMLF RW area failed. """
    pass
    
def setupConsoleLogging(*args, **kwargs):
    """ Initializes the logging of messages to the console. """
    framework.mmlf_logging.setupConsoleLogging(*args, **kwargs)

def setupFileLogging(logdir, *args, **kwargs):
    """ Initializes the logging of messages into a log file in *logdir*. """
    framework.mmlf_logging.setupFileLogging(logdir, *args, **kwargs)

def initializeRWArea(rwPath=None):
    """ Initialize the RW area.
    
    If *rwPath* is specified, the RW area located under that path is used.
    Otherwise, the standard RW area returned by *getRWPath()* is used.
    
    If the required RW area does not exist, it is automatically created under
    the specified path. If config files in the RW area are missing, they are
    automatically restored. 
    """
    # Set environment variable pointing to the MMLF RW directory
    if rwPath is None:
        rwPath = getRWPath()
    assert os.path.isabs(rwPath), \
                "Path to RW area must be absolute, not '%s'." % rwPath
    os.environ["MMLF_RW_PATH"] = rwPath
    log.info("Using MMLF RW area %s" % os.environ["MMLF_RW_PATH"])
    
    # Create directory for log files if not existing
    if not os.path.exists(rwPath + os.sep +  "logs"):
        os.makedirs(rwPath + os.sep +  "logs")    
    
    ## Create/Sync directory for config files
    # Find roPath
    if os.path.exists("config"): # local installation
        roPath = "config"
    elif sys.platform == 'win32': # global installation under Windows
        roPath = "%s\\mmlf\\config" % os.environ["APPDATA"]
    else : # global installation under unix-like OS
        roPath = "/etc/mmlf/config"
        
    # Copy config files from RO area to RW area
    for dirPath, dirNames, fileNames in os.walk(roPath):
        relPath = dirPath[len(roPath)+1:]
        for dir in dirNames:
            if not os.path.exists(os.path.join(rwPath, "config", relPath , dir)):
                os.makedirs(os.path.join(rwPath, "config", relPath , dir))
        for file in fileNames:
            if not os.path.exists(os.path.join(rwPath, "config", relPath, file)):
                log.info("MMLF RW Area: Creating %s" 
                             % os.path.join(rwPath, "config", relPath, file))         
                shutil.copy(os.path.join(dirPath, file), 
                            os.path.join(rwPath, "config", relPath, file))
    
def getRWPath():
    """ Return the path of the MMLF RW area to be used. 
    
    This defaults to the path given by the environment variable $MMLF_RW_PATH
    if this variable exists. Otherwise, $HOME/.mmlf is used if the environment
    variable $HOME exists (typically under unix-like OS). Under Windows OS,
    $USERPROFILE/.mmlf is used instead. If None of these variable exists,
    an Exception of type RWAreaInitializationFailedException is raised.
    """
    if "MMLF_RW_PATH" in os.environ:
        return os.environ["MMLF_RW_PATH"]
    elif "HOME" in os.environ:
        return os.environ["HOME"] + os.sep + ".mmlf"
    elif "USERPROFILE" in os.environ: # MS Windows
        return os.environ["USERPROFILE"]  + os.sep + ".mmlf"
    else:
        raise RWAreaInitializationFailedException(
                  "No RW area found. Please set environment variable MMLF_RW_PATH.")

def loadWorldFromConfigFile(configPath, useGUI=False):
    """ Load the world specified in the *configPath*.
    
    This method loads the MMLF world specified in the YAML-file under
    *configPath*. *configPath* can either be a relative path (in which case
    it is looked up in the config directory of MMLF RW area or an absolute path.
    The created MMLF World object is returned.
    
    *useGUI* indicates whether we are running a graphical user interface. If True,
    this causes the agent and environment of the world to create Viewers (and
    thus requires that PyQt4 is installed). 
    """
    from mmlf.framework.filesystem import BaseUserDirectory
    from mmlf.framework.world import World
    
    # Create base user directory
    baseUserDir = BaseUserDirectory()
    
    # Determine absoulte path of config file
    if not os.path.isabs(configPath):
        absConfigPath = baseUserDir.getAbsolutePath("rwconfig", configPath)
    else:
        absConfigPath = configPath
    
    # Load YAML-file
    worldConfigObject = yaml.load(open(absConfigPath, 'r'))
    
    # Load World
    return loadWorld(worldConfigObject, baseUserDir, useGUI=useGUI)
    

def loadWorld(worldConfigObject, baseUserDir=None, useGUI=False, 
              keepObservers=[]):
    """ Load the world specified in the *worldConfigObject*.
    
    *worldConfigObject* must be a dictionary defining the configuration of a world.
    This dictionary must have the following keys:
      :worldPackage: : The name of the subpackage of ``mmlf/world`` in which
                       the world is located.
      :environment: : Definition of the environment. Must be a dictionary with
                      the keys "moduleName" (the python module in which the 
                      environment is defined; looked up in the subdirectory
                      ``environments`` of the world package) and "configDict"
                      (the config dictionary of the environment, must have the 
                      same key-values as the environment's DEFAULT_CONFIG_DICT).
      :agents: : Definition of the agent. Must be a dictionary with the keys 
                 "moduleName" (the python module in which the agent is defined;
                 looked up in ``mmlf/agents`` or in the subdirectory ``agents``
                 of the world package) and "configDict" (the config dictionary
                 of the agent, must have the same key-values as the agent's 
                 DEFAULT_CONFIG_DICT).
      :monitor: : The configuration dictionary of the MMLF monitor.
    You may take a look at the 
    :ref:`Learn more about Experiments/Scripting experiments <scripted_experiments>`
    section of the MMLF documentation for an example. 
    
    *baseUserDir* must be a MMLF BaseUserDirectory object (which represent the
    MMLF RW area internally). If none, a new BaseUserDirectory object is created.
    
    *useGUI* indicates whether we are running a graphical user interface. If True,
    this causes the agent and environment of the world to create Viewers (and
    thus requires that PyQt4 is installed).
    
    Per default, loadWorld() removes all existing observer objects since these 
    stem from prior runs. Of observables should be kept, these can be passed via 
    the parameter *keepObservers* (a list).
    """
    from mmlf.framework.filesystem import BaseUserDirectory
    from mmlf.framework.world import World
    
    if baseUserDir == None:
        baseUserDir = BaseUserDirectory()
    
    # OBSERVABLES must contain only observables of type AllObservables and 
    # AllViewers when starting a world. Other observables are from prior runs 
    # and are deleted
    from mmlf.framework.observables import OBSERVABLES, AllObservables
    if useGUI:
        from mmlf.gui.viewers import AllViewers
    else:
        AllViewers= None
    cleanedListOfObservables = []
    for observable in OBSERVABLES.allObservables:
        if not (isinstance(observable, AllObservables) or (AllViewers
                  and isinstance(observable, AllViewers))):
            del(observable)
        else:
            observable.observers = filter(lambda o: o in keepObservers,
                                          observable.observers)
            cleanedListOfObservables.append(observable)
    OBSERVABLES.allObservables = cleanedListOfObservables
        
    # Create world object based on config object
    return World(worldConfigObject, baseUserDir, useGUI) 
    

def loadExperimentConfig(experimentPath):
    """ Load the specification of an MMLF experiment.
    
    Loads the specification of an MMLF experiment from the file
    ``experiment_config.yaml`` in *experimentPath*. Returns a dictionary 
    containing the experiment's specification.    
    """
    if not os.path.isabs(experimentPath):
        # Assume it is relative to root of MMLF RW-directory
        experimentPath = getRWPath() + os.sep + experimentPath
        
    experimentConfigFile = experimentPath + os.sep + "experiment_config.yaml"
    experimentConf = yaml.load(open(experimentConfigFile, 'r'))
        
    # Add all worlds specified in this directory to the world list
    experimentConf["worlds"] = dict()
    worldDir = experimentPath + os.sep + "worlds"
    for worldConfFile in glob.glob(worldDir + os.sep + "*.yaml"):
        worldConfigObject = yaml.load(open(worldConfFile, 'r'))
        worldName = worldConfFile.split(os.sep)[-1][:-5]
        experimentConf["worlds"][worldName] = worldConfigObject 
        
    return experimentConf

def runExperiment(experimentConf):
    """ Execute the experiment specified in experimentConf.
    
    *experimentConf* must be a dictionary defining the specification of the
    experiment. This dictionary must have the following keys:
      :concurrency: : "Concurrent" if several runs of the experiment should 
                      be executed at the same time, "Sequential" if one run 
                      after the other should be executed.
      :parallelProcesses: : If concurrent execution is requested, this parameter
                            specifies how many runs can be run in parallel. This
                            value is upper-bounded by the number of cores.
      :episodesPerRun: : Integer defining how many episodes each run of a world
                         should last.
      :runsPerWorld: : How many independent runs should be executed for each world.
      :worlds: : A dictionary mapping world name to a world configuration 
                 dictionary. See *loadWorld* for more details about world
                 configuration dictionaries.
    You may take a look at the 
    :ref:`Learn more about Experiments/Scripting experiments <scripted_experiments>`
    section of the MMLF documentation for an example. 
    """
    if sys.platform == "win32":
        if not os.path.exists(sys.argv[0] + ".py"):
            import shutil
            shutil.copyfile(sys.argv[0], sys.argv[0] + ".py")
    # Creating queues for inter-process communication
    from multiprocessing import Queue
    observableQueue = Queue()
    updateQueue = Queue()
    
    # Run the experiment in a separate thread
    from mmlf.framework.experiment import runExperiment
    from threading import Thread
    launcherThread = Thread(target=runExperiment, 
                            args=(experimentConf, observableQueue, 
                                  updateQueue, None, False))
    launcherThread.start()
    
    # Keep getting items from queues such that subprocesses can terminate
    while launcherThread.is_alive():
        try:
            observableQueue.get(False)
        except:
            pass
        try:
            updateQueue.get(False)
        except:
            pass
