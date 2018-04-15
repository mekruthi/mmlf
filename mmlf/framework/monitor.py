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
# Created: 2010/07/30
""" Monitoring of the MMLF based on logging performance metrics and graphics."""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import os
import inspect
import cPickle

from copy import deepcopy
import pylab

from mmlf.agents.agent_base import PolicyNotStorable

import mmlf.framework.mmlf_logging
from mmlf.framework.filesystem import LogFile
from mmlf.framework.observables import OBSERVABLES, FloatStreamObservable, \
                                    FunctionOverStateSpaceObservable, \
                                    StateActionValuesObservable, \
                                    ModelObservable

class Monitor(object):
    """ Monitor of the MMLF.
    
    The monitor supervises the execution of a world within the MMLF and stores
    certain selected information periodically. It always stores the values a
    FloatStreamObservable takes on into a file with the suffix "fso". For other 
    observables (FunctionOverStateSpaceObservable, StateActionValuesObservable,
    ModelObservable) a plot is generated and stored into files if this is 
    specified in the monitor's config dict (using functionOverStateSpaceLogging,
    stateActionValuesLogging, modelLogging).
    
    **CONFIG DICT** 
        :policyLogFrequency: : Frequency of storing the agent's policy in a serialized version to a file.
                               The policy is stored in the file policy_x in the 
                               subdirectory "policy" of the agent's log directory 
                               where x is the episodes' numbers. 
        :plotObservables: : The names of the observables that should be stored to a file. If "All", all observables are stored.
                            Defaults to "All" (also if plotObservables is not specified in a config file). 
        :stateActionValuesLogging: : Configuration of periodically plotting StateActionValuesObservables.
                                     Examples for StateActionValuesObservable are
                                     state-action value functions or stochastic policies.
                                     The plots are stored in the file episode_x.pdf in a 
                                     subdirectory  of the agent's log directory
                                     with the observable's name where x is the episodes' numbers.
        :functionOverStateSpaceLogging: : Configuration of periodically plotting FunctionOverStateSpaceObservables.
                                          Examples for FunctionOverStateSpaceObservable are
                                          state value functions or deterministic policies.
                                          The plots are stored in the file episode_x.pdf in a 
                                          subdirectory  of the agent's log directory
                                          with the observable's name where x is the episodes' numbers.
        :modelLogging: : Configuration of periodically plotting ModelObservables.
                         Examples for ModelObservables are models.
                         The plots are stored in the file episode_x.pdf in a 
                         subdirectory  of the agent's log directory
                         with the observable's name where x is the episodes' numbers.
        :active: : Whether the respective kind of logging is activated
        :logFrequency: : Frequency (in episodes) of creating a plot based on the respective observable and storing it to a file.
        :stateDims: : The state space dimensions that are varied in the plot. All other state space dimensions are kept constant.
                      If None, the stateDims are automatically deduced. This is only possible under specific conditions (2d state space) 
        :actions: : The actions for which separate plots are created for each StateActionValuesObservable.
        :rasterPoints: : The resolution (rasterPoint*rasterPoint) of the plot in continuous domain.
        :colouring: : The background colouring of a model plot. Can be either "Rewards" or "Exploration".
                      If "Rewards", the reward predicted by the model is used for 
                      as background, while for "Exploration", each state-action 
                      pair tried at least minExplorationValue times is coloured
                      with one colour, the others with an other colour.
        :plotSamples: : If true, the state-action pairs that have been observed are plotted into the model-plot. Otherwise the model's predictions. 
        :minExplorationValue: : If the colouring is "Exploration", each state-action pair tried at least minExplorationValue times is coloured with one colour, the others with an other colour.        
    """
    
    DEFAULT_CONFIG_DICT = {'policyLogFrequency' : 1,
                           'plotObservables' : "All",
                           'stateActionValuesLogging' :
                              {'active' : False,
                               'logFrequency' : 10,
                               'stateDims': None,
                               'actions' : None,
                               'rasterPoints' : 50},
                           'functionOverStateSpaceLogging':
                               {'active' : False,
                                'logFrequency' : 10,
                                'stateDims' : None,
                                'rasterPoints' : 50},
                            'modelLogging':
                               {'active' : False,
                                'logFrequency' : 10,
                                'colouring' : "Rewards",
                                'plotSamples' : True,
                                'minExplorationValue' : 1,
                                'stateDims' : None,
                                'rasterPoints' : 25}}
    
    def __init__(self, world, configDict):
            
        # Recursively evaluate config dict 
        def evaluateConfigDict(partConfigDict):
            evalConfigDict = {}
            for key, value in partConfigDict.iteritems():
                if isinstance(value, dict):
                    evalConfigDict[key] = evaluateConfigDict(value)                    
                elif isinstance(value, basestring):
                    try:
                        evalConfigDict[key] = eval(value)
                    except NameError:
                        import warnings
                        warnings.warn("Value %s for key %s of monitor config can not "
                                      "be evaluated." % (value, key))
                        evalConfigDict[key] = value
                else:
                    evalConfigDict[key] = value
                    
            return evalConfigDict
        
        self.configDict = evaluateConfigDict(configDict)
        
        # Determine environment and agent
        self.environment = world.environment
        self.agent = world.agent
        
        self.episodeCounter = 0
        
        # Setup logging       
        self.log = mmlf.framework.mmlf_logging.getLogger('MonitorLog')
        
        self.userDirObj = world.baseUserDir
        self.userDirObj.createPath(['Monitor'], refName='monitorlogs', 
                                   baseRef='currentlogdir', force=True)
        self.userDirObj.createPath(['%s' % self.agent.__class__.__name__], 
                                   refName='agentlogs', baseRef='currentlogdir',
                                   force=True)
        self.userDirObj.createPath(['%s' % self.environment.__class__.__name__], 
                                   refName='environmentlogs', baseRef='currentlogdir',
                                   force=True)
        self.userDirObj.createPath(['policy'], refName='policydir',
                                   baseRef='agentlogs', force=True)
        
        # Create function that determines whether a certain observable should
        # be plotted
        if not 'plotObservables' in self.configDict \
                or self.configDict['plotObservables'] == "All" \
                or "All" in self.configDict['plotObservables']:
            self.plotThisObservableFct = lambda observableName: True
        else:
            self.plotThisObservableFct = \
                lambda observableName: observableName in self.configDict['plotObservables']
        
        # Logging of float stream observables to files
        self._initializeFileLogging()
    
        # Logging graphics            
        if "stateActionValuesLogging" in self.configDict \
            and self.configDict["stateActionValuesLogging"]["active"]:
            self._initializeStateActionValuesLogging(self.configDict["stateActionValuesLogging"])
            
        if "functionOverStateSpaceLogging" in self.configDict \
            and self.configDict["functionOverStateSpaceLogging"]["active"]:
            self._initializeFunctionOverStateSpaceLogging(self.configDict["functionOverStateSpaceLogging"])
            
        self._initializeModelLogging(self.configDict.get('modelLogging', None))
            
        if "compareToOptimalPolicy" in self.configDict and self.configDict["compareToOptimalPolicy"]:
            self.optimalPolicyPath = self._getOptimalPolicyPath()
            self.optimalPolicyAvailable = os.path.exists(self.optimalPolicyPath)
            
            self.optimalPolicyComputable = hasattr(self.environment, 
                                                   "computeOptimalPolicy")
            self.optimalPolicy = None

            self.correctDecisionLog = LogFile(self.userDirObj,
                                              'correctDecisionRatio',
                                              baseRef='monitorlogs')
            
    def notifyEndOfEpisode(self):
        """ Notify monitor that an episode in the world has terminated. """
        # If policy log frequency is not specified, we set it to 1.
        if not 'policyLogFrequency' in self.configDict:
            self.configDict['policyLogFrequency'] = 1
            
        if (self.episodeCounter + 1) % self.configDict['policyLogFrequency'] == 0:
            # Try to log policy
            try:
                filePath = \
                    self.userDirObj.getAbsolutePath('policydir',
                                                    'policy_' + str(self.episodeCounter))
                self.agent.storePolicy(filePath, optimal=True)
            except PolicyNotStorable, e:
                import warnings
                warnings.warn("The greedy policy could not be stored. "
                              "Exception: %s" % e) 
               
        if "compareToOptimalPolicy" in self.configDict and self.configDict["compareToOptimalPolicy"]:
            self._compareToOptimalPolicy()
            
        self.episodeCounter += 1
        
    def _initializeFileLogging(self):
        """ Initialize logging of FloatStreamObservables into files. """
        # Helper function that handles newly added FloatStreamObservables
        def addRemoveObservable(observable, action):
            # Check if new observable was added and if the observable is a 
            # FloatStreamObservable
            if action == 'added' and isinstance(observable, FloatStreamObservable):
                # Register function that writes all updates into log file
                logDirectory = observable.title.split(" ")[0]
                logFileName = "".join(observable.title.split(" ")[1:])
                logFileName += ".fso" # FloatStreamObservable
                self.userDirObj.createPath([logDirectory], refName=logDirectory, 
                                           baseRef='currentlogdir', force=True)
                logFile = LogFile(self.userDirObj, logFileName, 
                                  baseRef=logDirectory)
                def updateLogFile(time, value, *args):
                    logFile.addText("%s\t%s\n" % (time, value))
                observable.addObserver(updateLogFile)
        
        # For all FloatStreamObservable that have been created already,
        # call fiunction addRemoveObservable
        for observable in OBSERVABLES.getAllObservablesOfType(FloatStreamObservable):
            addRemoveObservable(observable, 'added')
                
        # Observe set of FloatStreamObservable    
        OBSERVABLES.addObserver(addRemoveObservable)
    
    def _initializeStateActionValuesLogging(self, spec):
        """ Initialize logging of state action values into files. """       
        # Helper function that handles newly added StateActionValuesObservable
        def addRemoveObservable(observable, action):
            # Check if new observable was added and if the observable is a 
            # StateActionValuesObservable
            if action == 'added' and isinstance(observable, 
                                                StateActionValuesObservable):
                # Register function that writes all updates into log file
                topDirectory = observable.title.split(" ")[0]
                logDirectory = "".join(observable.title.split(" ")[1:]).strip("()")
                if not self.plotThisObservableFct(logDirectory):
                    # Plotting this observable was not specified
                    return
                self.userDirObj.createPath([topDirectory], refName=topDirectory, 
                                           baseRef='currentlogdir', force=True)
                self.userDirObj.createPath([logDirectory], refName=logDirectory, 
                                           baseRef=topDirectory, force=True)
                def plotGraphicToFile(function, actions):
                    if (self.episodeCounter + 1) % spec['logFrequency'] != 0:
                        return # Not an episode in whcih we plot
                    logFile = \
                        self.userDirObj.getAbsolutePath(logDirectory,
                                                        'episode_%05d.pdf' % self.episodeCounter)
                    if os.path.exists(logFile):
                        return # we do not plot several times per episode
                    fig = pylab.figure(0, figsize = (22,11))
                    fig.clear()
                              
                    # Let observable plot itself          
                    observable.plot(function, actions, fig,
                                    self.environment.stateSpace,
                                    plotStateDims=spec['stateDims'],
                                    plotActions=spec['actions'],
                                    rasterPoints=spec['rasterPoints'])
                    
                    # Draw structure over state space
                    for axis in fig.axes:
                        if type(axis) == __import__("matplotlib").axes.Axes:
                            # Plot only into matplotlib.axes.AxesSubplot
                            continue
                        self.environment.plotStateSpaceStructure(axis)

                    pylab.savefig(logFile)
                    
                observable.addObserver(plotGraphicToFile)
        
        # For all StateActionValuesObservable that have been created already,
        # call function addRemoveObservable
        for observable in OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable):
            addRemoveObservable(observable, 'added')
            
        # Observe set of StateActionValuesObservable    
        OBSERVABLES.addObserver(addRemoveObservable)
                
    def _initializeFunctionOverStateSpaceLogging(self, spec):
        """ Initialize logging of function over state spaces into files. """
        # Helper function that handles newly added FunctionOverStateSpaceObservable
        def addRemoveObservable(observable, action):
            # Check if new observable was added and if the observable is a 
            # FunctionOverStateSpaceObservable
            if action == 'added' and isinstance(observable, 
                                                FunctionOverStateSpaceObservable):
                # Register function that writes all updates into log file
                topDirectory = observable.title.split(" ")[0]
                logDirectory = "".join(observable.title.split(" ")[1:]).strip("()")
                if not self.plotThisObservableFct(logDirectory):
                    # Plotting this observable was not specified
                    return
                self.userDirObj.createPath([topDirectory], refName=topDirectory, 
                                           baseRef='currentlogdir', force=True)
                self.userDirObj.createPath([logDirectory], refName=logDirectory, 
                                           baseRef=topDirectory, force=True)
                def plotGraphicToFile(function):
                    if (self.episodeCounter + 1) % spec['logFrequency'] != 0:
                        return # Not an episode in whcih we plot
                    logFile = \
                        self.userDirObj.getAbsolutePath(logDirectory,
                                                        'episode_%05d.pdf' % self.episodeCounter)
                    if os.path.exists(logFile):
                        return # we do not plot several times per episode
                    fig = pylab.figure(0, figsize = (22,11))
                    fig.clear()
                                        
                    # Let observable plot itself
                    observable.plot(function, fig, 
                                    stateSpace=self.environment.stateSpace, 
                                    actionSpace=self.environment.actionSpace, 
                                    plotStateDims=spec['stateDims'],
                                    rasterPoints=spec['rasterPoints'])
                    
                    # Draw structure over state space
                    for axis in fig.axes:
                        if type(axis) == __import__("matplotlib").axes.Axes:
                            # Plot only into matplotlib.axes.AxesSubplot
                            continue
                        self.environment.plotStateSpaceStructure(axis)

                    pylab.savefig(logFile)
                    
                observable.addObserver(plotGraphicToFile)
        
        # For all FunctionOverStateSpaceObservable that have been created already,
        # call function addRemoveObservable
        for observable in OBSERVABLES.getAllObservablesOfType(FunctionOverStateSpaceObservable):
            addRemoveObservable(observable, 'added')
            
        # Observe set of FunctionOverStateSpaceObservable    
        OBSERVABLES.addObserver(addRemoveObservable)


    def _initializeModelLogging(self, spec):
        """ Initialize logging of function over state spaces into files. """
        # Helper function that handles newly added ModelObservable
        def addRemoveObservable(observable, action):
            # Check if new observable was added and if the observable is a 
            # ModelObservable
            if action == 'added' and isinstance(observable, 
                                                ModelObservable):
                from mmlf.gui.viewers import VIEWERS
                from mmlf.gui.viewers.model_viewer import ModelViewer
                if spec is not None and spec["active"]:
                    # Register function that writes all updates into log file
                    topDirectory = observable.title.split(" ")[0]
                    logDirectory = "".join(observable.title.split(" ")[1:]).strip("()")
                    if not self.plotThisObservableFct(logDirectory):
                        # Plotting this observable was not specified
                        return
                    self.userDirObj.createPath([topDirectory], refName=topDirectory, 
                                               baseRef='currentlogdir', force=True)
                    self.userDirObj.createPath([logDirectory], refName=logDirectory, 
                                               baseRef=topDirectory, force=True)
                    def plotGraphicToFile(model):
                        if (self.episodeCounter + 1) % spec['logFrequency'] != 0:
                            return # Not an episode in which we plot
                        logFile = \
                            self.userDirObj.getAbsolutePath(logDirectory,
                                                            'episode_%05d.pdf' % self.episodeCounter)
                        if os.path.exists(logFile):
                            return # we do not plot several times per episode
                        fig = pylab.figure(0, figsize = (22,11))
                        fig.clear()
                                            
                        # Let observable plot itself
                        observable.plot(model, fig, 
                                        stateSpace=self.environment.stateSpace,
                                        colouring=spec['colouring'],
                                        plotSamples=spec['plotSamples'], 
                                        minExplorationValue=spec['minExplorationValue'],
                                        plotStateDims=spec['stateDims'],
                                        dimValues=spec['rasterPoints'])
                        
                        # Draw structure over state space
                        for axis in fig.axes:
                            if type(axis) == __import__("matplotlib").axes.Axes:
                                # Plot only into matplotlib.axes.AxesSubplot
                                continue
                            self.environment.plotStateSpaceStructure(axis)
    
                        pylab.savefig(logFile)
                    
                    observable.addObserver(plotGraphicToFile)
                
                def createModelViewer():
                    modelViewer = ModelViewer(observable,
                                              self.environment.stateSpace)    
                    observable.addObserver(lambda *_x: modelViewer.update(*_x))
                    return modelViewer
                    
                VIEWERS.addViewer(createModelViewer, 'ModelViewer')
        
        # For all ModelObservable that have been created already,
        # call function addRemoveObservable
        for observable in OBSERVABLES.getAllObservablesOfType(ModelObservable):
            addRemoveObservable(observable, 'added')
            
        # Observe set of ModelObservable
        OBSERVABLES.addObserver(addRemoveObservable)

    
    def _compareToOptimalPolicy(self):
        # Check if environment supports computation of optimal policy
        if not hasattr(self.environment, "getStateSet"): return 
        if self.optimalPolicy == None:
            # Compute optimal policy
            self.stateSet = self.environment.getStateSet()
            
            if self.optimalPolicyAvailable:
                self.log.info("Loading cached optimal policy.")
                optimalPolicyFile = open(self.optimalPolicyPath, 'r')
                self.optimalPolicy = cPickle.load(optimalPolicyFile)
                optimalPolicyFile.close()            
            elif self.optimalPolicyComputable:
                self.log.info("Computing optimal policy in environment...")
                self.optimalPolicy = self.environment.computeOptimalPolicy()
                self.log.info("Computing optimal policy in environment... Done.")                
                
                self.log.info("Storing optimal policy for later reuse.")
                cacheDir = os.sep.join(self.optimalPolicyPath.split(os.sep)[:-1])
                if not os.path.exists(cacheDir):
                    os.makedirs(cacheDir)
                optimalPolicyFile = open(self.optimalPolicyPath, 'w')
                cPickle.dump(self.optimalPolicy, optimalPolicyFile)
                optimalPolicyFile.close()        

        agentPolicy = self.agent.getGreedyPolicy()
        
        stateCounter = 0
        correctCounter = 0
        for state in self.stateSet:
            correctCounter += (agentPolicy.evaluate(state)[0] 
                                    in self.optimalPolicy.getOptimalActions(state))
            stateCounter += 1
        
        correctDecisionRatio = float(correctCounter) / stateCounter
        
        self.correctDecisionLog.addText("%s\n" % correctDecisionRatio)
                
     
    def _getOptimalPolicyPath(self):
        environmentConfigHash = abs(hash(str(self.environment.configDict)))
        
        envModuleFile = inspect.getsourcefile(self.environment.__class__)
        
        policyPath = os.sep.join(envModuleFile.split(os.sep)[:-1]) + os.sep \
                        + "policy_cache" + os.sep  \
                        + "%s.pickle" % environmentConfigHash

        return policyPath   
        
