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
# Created: 2009/07/20
""" The Model-based Direct Policy Search agent
   
This module contains an agent that uses the state-action-reward-successor_state
transitions to learn a model of the environment. It performs than direct policy
search (similar to the direct policy search agent using a black-box optimization
algorithm to optimize the parameters of a parameterized policy) in the model
in order to optimize a criterion defined by a fitness function. This fitness
function can be e.g. the estimated accumulated reward obtained by this
policy in the model environment. In order to enforce exploration, the model
is wrapped for an RMax-like behavior so that it returns the reward RMax for all
states that have not been sufficiently explored.  
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import copy
import numpy
#import pylab

import mmlf.framework.protocol
from mmlf.agents.agent_base import AgentBase
from mmlf.framework.observables import FloatStreamObservable
from mmlf.agents.dps_agent import DPS_Agent
from mmlf.resources.model.model import Model, ModelNotInitialized
from mmlf.resources.model.rmax_model_wrapper import RMaxModelWrapper
from mmlf.resources.planner.mbdps_planner import MBDPSPlanner, estimatePolicyOutcome

class MBDPS_Agent(DPS_Agent):
    """ The Model-based Direct Policy Search agent
    
    An agent that uses the state-action-reward-successor_state transitions to 
    learn a model of the environment. It performs direct policy search
    (similar to the direct policy search agent using a black-box optimization
    algorithm to optimize the parameters of a parameterized policy) in the model
    in order to optimize a criterion defined by a fitness function. This fitness
    function can be e.g. the estimated accumulated reward obtained by this
    policy in the model environment. In order to enforce exploration, the model
    is wrapped for an RMax-like behavior so that it returns the reward RMax for 
    all states that have not been sufficiently explored. RMax should be an upper
    bound to the actual achievable return in order to enforce optimism in the
    face of uncertainty.
       
    **CONFIG DICT** 
        :gamma: : The discount factor for computing the return given the rewards
        :planning_episodes: : The number internally simulated episodes that are performed in one planning step 
        :policy_search: : The method used for search of an optimal policy in the policy space. Defines policy parametrization and internally used black box optimization algorithm.
        :model: : The algorithm used for learning a model of the environment                 
    """
    
    DEFAULT_CONFIG_DICT = {'gamma' : 1.0,
                           'planning_episodes' : 1000,
                           'policy_search' : {'method': 'fixed_parametrization',
                                              'policy': {'type': 'linear',
                                                         'bias': True,
                                                         'numOfDuplications': 1},
                                              'optimizer': {'name': 'evolution_strategy', 
                                                            'sigma':  1.0,
                                                            'evalsPerIndividual': 10,
                                                            'populationSize' : 5,
                                                            'numChildren' :10}},
                            'model' :  {'name' : 'RMaxModelWrapper',
                                        'model' : {'name': 'KNNModel',
                                                   'k': 100,
                                                   'b_Sa': 0.03,
                                                   'exampleSetSize': 2500},
                                        'RMax' : 0.0,
                                        'minExplorationValue' : 1.0}}
    
    def __init__(self, *args, **kwargs):
        
        # Create the agent info
        self.agentInfo = mmlf.framework.protocol.AgentInfo(
                            versionNumber = "0.3",
                            agentName= "Model-based Direct Policy Search",
                            continuousState = True,
                            continuousAction = False,
                            discreteAction = True,
                            nonEpisodicCapable = True)
        
        super(MBDPS_Agent, self).__init__(*args, **kwargs)
                            
        # We perform the repeated evaluation of an individual internally to
        # have control over the start states
        self.evalsPerIndividual = self.configDict["policy_search"]["optimizer"]["evalsPerIndividual"]
        self.configDict["policy_search"]["optimizer"]["evalsPerIndividual"] = 1
        
        self.startState = None

        self.maxEpisodeLength = 1
        
        self.lastExplorationValue = 0
        
        self.bestPolicy = None
        self.maxReturn = -numpy.inf
        
        self.returnEstimate = None
        self.estimatedTrajectory = None
        
        # In certain situations (unitialized model, unexplored regions), we act
        # randomly
        self.actRandomly = True
        
        # Prepare logging
        self.userDirObj.createPath(['model'], refName='modelDir',
                                   baseRef='agentlogs', force=True)
        self.userDirObj.createPath(['actionModel'], refName='actionModelDir',
                                   baseRef='modelDir', force=True)
        self.userDirObj.createPath(['trajectory_deviation'],
                                   refName='trajectoryDeviationDir',
                                   baseRef='agentlogs', force=True)
        self.userDirObj.createPath(['trajectory3d'], refName='trajectory3dDir',
                                   baseRef='agentlogs', force=True)
#        self.userDirObj.createPath(['transitions'],
#                                   refName='transitionDir',
#                                   baseRef='agentlogs',
#                                   force=True)          
        # An observable that allows to observe the return prediction
        self.returnPredictionObservable = \
            FloatStreamObservable(title='%s Exploration Value' % self.__class__.__name__,
                                  time_dimension_name='Episode',
                                  value_name='Exploration Value')
        #  An observable that allows to observe the exploration value
        self.explorationValueObservable = \
            FloatStreamObservable(title='%s Exploration Value' % self.__class__.__name__,
                                  time_dimension_name='Step',
                                  value_name='Exploration Value')
     
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
        
    def giveReward(self, reward):
        """ Provides a reward to the agent """
        self.reward = reward
        self.accumulatedReward += reward
    
    def getAction(self):
        """ Request the next action the agent want to execute """
        # Remember the maximum length of an episode (as an upper limit for
        # planning episodes length)
        self.maxEpisodeLength = max(self.maxEpisodeLength, self.stepCounter)
        
        
        # Update model    
        if self.lastState != None:       
            # Inform the model about the outcome of the last action
            self.model.addExperience(self.lastState, tuple(self.lastAction),
                                     self.state, self.reward)
        else:
            # Remember the start state of the episode
            self.startState = self.state
            # Inform the model about a new start state 
            self.model.addStartState(self.startState)

        # Check if we have to replan
        doReplanning, localReplanning = self._doReplanning()
        if doReplanning:
            self._optimizePolicy(localReplanning)
#            import cProfile; 
#            cProfile.runctx("self._optimizePolicy(self.state)", globals(), 
#                            locals(), "profiling_data.dat")

        # The concrete action can be chosen as in the DPS agent
        if not self.actRandomly:
            actionTaken = super(MBDPS_Agent, self).getAction()
        else:
            # Act randomly
            actionDictionary = self.actionSpace.sampleRandomAction()
            randomAction = []
            for index, actionName in enumerate(self.actionSpace.iterkeys()):
                randomAction.append(actionDictionary[actionName])
                
            actionTaken = self._generateActionObject(actionDictionary)
            
            self.action = tuple(randomAction)
            # This sets self.lastAction = self.action and self.action to None  
            AgentBase.getAction(self)
            
        # Compute and store the immediate exploration value obtained for the 
        # given action in the given state in a file
        explorationValue = self.model.getExplorationValue(self.lastState,
                                                          self.lastAction)
        if explorationValue < self.model.minExplorationValue <= self.lastExplorationValue:
            self.agentLog.info("Step: %s. Entered unexplored region. " 
                               % self.stepCounter)
        if explorationValue >= self.model.minExplorationValue > self.lastExplorationValue:
            self.agentLog.info("Step: %s. Left unexplored region. " 
                               % self.stepCounter)
        self.lastExplorationValue = explorationValue
        # Update explorationValueObservable
        self.explorationValueObservable.addValue(self.stepCounter,
                                                 self.lastExplorationValue)
        
        return actionTaken
   
    def nextEpisodeStarted(self, terminalStateReached=True):
        """ Informs the agent that a new episode has started."""
        # If the agent has actually reached a terminal state
        if self.state is not None:
            # Inform the model about the outcome of the last action
            self.model.addExperience(self.lastState, tuple(self.lastAction), 
                                     self.state, self.reward)
            # We have reached a terminal state
            self.model.addTerminalState(self.state)            
                    
#        # Plotting
#        if self.plottingActive:
#            self._plot(rasterPoints = self.configDict["plotting"]["policyRasterPoints"])
#            
#            self.trajectories.append([])
#            self.trajectories = self.trajectories[-50:]
#            self.actionSequence = []
#            
#        if terminalStateReached:
#            # Create file to which the graphic is stored
#            transitionFileName = self.userDirObj.getAbsolutePath('transitionDir',
#                                                      ('%6d_transition.pickle' 
#                                                        % (self.episodeCounter)).replace(' ', '0'))
#            transitionFile = open(transitionFileName, 'w')
#            import cPickle
#            cPickle.dump(self.model.actionModels.values()[0].exampleSet, 
#                         transitionFile)
#            transitionFile.close()

        # Remember the optimal policy found so far
        if self.accumulatedReward > self.maxReturn:
            self.bestPolicy = copy.copy(self.policy)
            self.maxReturn = self.accumulatedReward 
            
        # Update returnPredictionObservable
        self.returnPredictionObservable.addValue(self.episodeCounter,
                                                 self.returnEstimate - self.accumulatedReward)
             
        super(MBDPS_Agent, self).nextEpisodeStarted(doPolicySearch=False)

    ######################  End COMMAND-HANDLING METHODS #######################

    def getGreedyPolicy(self):
        """ Returns the optimal policy the agent has found so far """
        return self.bestPolicy

    def _initialize(self):
        """ Lazy initialization of the agent once state and action space are known """
        super(MBDPS_Agent, self)._initialize()
    
        # Choose action model class based on conf
        self.model = Model.create(self.configDict["model"], self, self.stateSpace,
                                  self.actionSpace, self.userDirObj) 
        
        # Create planner
        self.planner = MBDPSPlanner(gamma=self.configDict["gamma"], 
                                    planningEpisodes=self.configDict["planning_episodes"],
                                    evalsPerIndividual=self.evalsPerIndividual)
    
    def _doReplanning(self):
        """ Decide whether we should adjust the policy. 
        
        Return a pair of booleans, the first one indicating whether we should
        replan (i.e. adjust the policy) and the second one whether the policy
        should be optimized for the current state (e.g. during exploration 
        phases) or for the whole start state distribution.
        """        
        if self.lastState == None:
            # Replan starting from current state at the start of an episode      
            self.agentLog.info("Planning at the start of a new episode.")
            return True, False
        elif self.actRandomly:
            # Check whether we are in an unexplored region where we should still
            # act randomly 
            explorationValue = \
                    min(self.model.getExplorationValue(self.state, action)
                            for action in self.actionSpace.getActionList())
            if explorationValue < self.model.minExplorationValue:
                # Continue to act randomly
                return False, False
            else:
                # We stop exploring and have to adjust the policy
                self.agentLog.info("Step: %s. Left unexplored region. Stop "
                                   "acting randomly. Replan." % self.stepCounter)
                self.actRandomly = False
                return True, True
        elif hasattr(self.model, "minExplorationValue"):
            # Check if we left an unexplored region and should thus not follow the
            # exploration policy any longer
            policyAction = self.policy.evaluate(self.state)
            policyActionExplorationValue = \
                self.model.getExplorationValue(self.state, policyAction)
            if self.lastExplorationValue < self.model.minExplorationValue <= policyActionExplorationValue:
                # Check if we could choose an action that is underexplored 
                # If yes, we start choosing unexplored actions randomly.
                explorationValue = \
                    min(self.model.getExplorationValue(self.state, action)
                            for action in self.actionSpace.getActionList())
                if explorationValue  < self.model.minExplorationValue <= policyActionExplorationValue:
                    self.agentLog.info("Step: %s. Start acting randomly." 
                                        % self.stepCounter)
                    self.actRandomly = True
                    return False, False
                else:
                    self.agentLog.info("Step: %s. Policy would leave unexplored "
                                       "region. Replan." % self.stepCounter)
                    return True, True
        # Do not replan
        return False, False        
    
    def _optimizePolicy(self, localReplanning):
        """ Determine the policy for this episode """
        # Reset policy search method
        self.policySearch.reset()

        # Use planner to conduct policy search based on fitness obtained from
        # sampling trajectories in the model
        if localReplanning:
            # Find only a policy that is good in the local region around the 
            # current state (used for exploration)
            self.policy, self.actRandomly, policyChanged =\
                    self.planner.plan(self.policy, self.policySearch, self.model,
                                      maxEpisodeLength=30,
                                      startState=self.state)
        else:
            # Find a policy that is suitable for the start state distribution 
            self.policy, self.actRandomly, policyChanged =\
                    self.planner.plan(self.policy, self.policySearch, self.model,
                                      maxEpisodeLength=self.maxEpisodeLength,
                                      startState=None)
            
        
        if policyChanged:
            # Store the parameters of the policy used in this episode
            self.policyParameterLog.addText("%s\n" % (self.policy.getParameters()))
    
        try:
            # Compute the estimated return and the trajectory that would 
            # have resulted from the start state when following the same 
            # policy in the model as in reality.
            model = self.model.model if isinstance(self.model, RMaxModelWrapper) \
                            else self.model                            
            self.returnEstimate, self.estimatedTrajectory = \
                 estimatePolicyOutcome(self.state, model, self.policy,
                                       self.maxEpisodeLength, self.configDict["gamma"],
                                       returnTrajectory=True)
        except ModelNotInitialized:
            # Model not initialized, set the estimated return to RMax
            # and the estimate trajectory to None
            self.returnEstimate = self.configDict["model"]["RMax"]
            self.estimatedTrajectory = None        
    
    ######################  Plotting #######################
         
#    def _plot(self, trajectories=None, plotStateDims=None, rasterPoints=100):
#        """ Plot the current policy and the current model """ 
#        if (self.episodeCounter + 1) % self.plottingFrequency == 0:           
#            # Plot the simulated and the actual trajectory and the policy
#            plotTrajectories = {'actual': self.trajectories[-1],
#                                'estimated': self.estimatedTrajectory}
#            super(MBDPS_Agent, self)._plot(rasterPoints=rasterPoints,
#                                           trajectories=plotTrajectories)            
#            
#            # Plot the current model
#            try:                
#                self.model.plot(self.stateSpace,
#                                self.configDict["plotting"]["plotStateDims"],
#                                self.configDict["plotting"]["modelRasterPoints"],
#                                self.configDict["model"].get("min_exploration_value",0),
#                                self.episodeCounter)
#                
#                # Create dim value versus time plots of actual and estimated 
#                # trajectories
#                self._plotTrajectoryDeviation()
#               
#                # For less than 5 state dimensions and more than two dimensions:
#                if 2 < len(self.stateSpace) <= 4:
#                    # Create 3d plots of all triples of state space dimensions 
#                    # of the actual trajectory and the action taken in each step
#                    self._plotTrajectory3d()
#            except ModelNotInitialized, e:
#                pass
#
#    def _plotTrajectoryDeviation(self):
#        """ Plot the development of the estimated and actual trajectory over time """
#        if self.estimatedTrajectory == None: 
#            return
#        
#        numStateDimensions = len(self.stateSpace.keys())
#        
#        # For each dimension
#        for dimIndex in range(numStateDimensions):
#            pylab.subplot(int(numpy.ceil((numStateDimensions + 1)/2.0)),
#                          int(numpy.ceil(numStateDimensions/2.0)),
#                          dimIndex + 1)
#            
#            # Plot actual trajectory
#            y = map(lambda t: t[dimIndex], self.trajectories[-1])
#            x = range(len(y))
#            pylab.plot(x, y, c='r', label='actual')
#            
#            # Plot estimated trajectory
#            y = map(lambda t: t[dimIndex], self.estimatedTrajectory)
#            x = range(len(y))
#            pylab.plot(x, y, c='b', label='estimated')
#            
#            pylab.ylim(0, 1)
#            pylab.xlabel("step")
#            pylab.ylabel(self.stateSpace.keys()[dimIndex])
#            pylab.legend()
#            
#        logFile = self.userDirObj.getAbsolutePath('trajectoryDeviationDir',
#                                                      '%06d_trajectory2d.png' 
#                                                         % self.episodeCounter)
#        pylab.savefig(logFile)
#        pylab.close()
#        
#    
#    def _plotTrajectory3d(self):
#        """ Plots the trajectory obtained in the episode
#        
#        Plots the trajectory obtained in the episode for all triples of state 
#        space dimensions. 
#        """     
#        for dimIndex1 in range(len(self.stateSpace.keys())):
#            for dimIndex2 in range(dimIndex1 + 1, 
#                                   len(self.stateSpace.keys())):
#                for dimIndex3 in range(dimIndex2 + 1,
#                                       len(self.stateSpace.keys())):
#                    logFile = self.userDirObj.getAbsolutePath('trajectory3dDir',
#                                                              '%06d_trajectory3d_%s_%s_%s.png' 
#                                                                 % (self.episodeCounter,
#                                                                    dimIndex1, dimIndex2,
#                                                                    dimIndex3))
#                    x = map(lambda x: x[dimIndex1], self.trajectories[-1])
#                    y = map(lambda x: x[dimIndex2], self.trajectories[-1])
#                    z = map(lambda x: x[dimIndex3], self.trajectories[-1])
#                    
#                    from mpl_toolkits.mplot3d import Axes3D
#                    fig = pylab.figure()
#                    ax = Axes3D(fig)
#
#                    ax.scatter(x, y, z)
#                    if not self.actionSpace.hasContinuousDimensions() and len(self.actions) <= 3:
#                        colors = dict(zip(self.actions, ['r', 'b', 'g']))
#                        for i, action in enumerate(self.actionSequence):
#                            ax.plot(x[i:i+2],y[i:i+2],z[i:i+2], c=colors[action])
#                    else:
#                        ax.plot(x, y, z, c='r')
#                    
#                    ax.set_xlim3d(0, 1)
#                    ax.set_ylim3d(0, 1)
#                    ax.set_zlim3d(0, 1)
#                    
#                    ax.set_xlabel(self.stateSpace.keys()[dimIndex1])
#                    ax.set_ylabel(self.stateSpace.keys()[dimIndex2])
#                    ax.set_zlabel(self.stateSpace.keys()[dimIndex3])
#
#                    pylab.savefig(logFile)
#                    pylab.close()
    
AgentClass = MBDPS_Agent
AgentName = "Model-based Direct Policy Search"
