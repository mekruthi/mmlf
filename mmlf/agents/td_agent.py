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
# Created: 2007/08/16
""" Agent based on temporal difference learning

This module defines a base agent for all kind of agents based on 
temporal difference learning. Most of these agents can reuse most methods
of this agents and have to modify only small parts. 

Note: The TDAgent cannot be instantiated by itself, it is a abstract base class!
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random

from mmlf.agents.agent_base import AgentBase

from mmlf.resources.learning_algorithms.temporal_difference import TD_Learner
from mmlf.resources.function_approximators.function_approximator \
        import FunctionApproximator
from mmlf.resources.policies.value_function_policy import ValueFunctionPolicy
from mmlf.framework.observables import StateActionValuesObservable, \
                                    FunctionOverStateSpaceObservable

class TDAgent(AgentBase):
    """
    A base agent for all kind of agents based on temporal difference learning.
    Most of these agents can reuse most methods of this agents and 
    have to modify only small parts
    
    Note: The TDAgent cannot be instantiated by itself, it is a abstract base class!
    """
    
    def __init__(self, *args, **kwargs):
        
        super(TDAgent, self).__init__(*args, **kwargs)
        
        self.epsilon = self.configDict.get('epsilon', 0.0)
        self.epsilonDecay = self.configDict.get('epsilonDecay', 1.0)
                       
        # An observable that can be used to monitor an agents greedy policy
        self.policyObservable = \
            FunctionOverStateSpaceObservable(title='%s (greedy policy)' % self.__class__.__name__,
                                             discreteValues=True)
        # An observable that can be used to monitor an agents optimal value function
        self.optimalValueFunctionObservable = \
            FunctionOverStateSpaceObservable(title='%s (optimal value function)' % self.__class__.__name__,
                                                 discreteValues=False)
        # An observable that can be used to monitor an agents Q-Function
        self.stateActionValuesObservable = \
            StateActionValuesObservable(title='%s (value function)' % self.__class__.__name__)
     
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
    
    def setStateSpace(self, stateSpace):
        """ Informs the agent about the state space of the environment
        
        More information about state spaces can be found in 
        :ref:`state_and_action_spaces`
        """ 
        super(TDAgent, self).setStateSpace(stateSpace)
        
        # Determine the resolution used for continuous state space dimensions
        # (if any exist):
        for dimName, dimDescr in self.stateSpace.iteritems():
            if dimDescr['dimensionType'] == "continuous": # Set the resolution
                if isinstance(self.configDict['stateDimensionResolution'], dict):
                    # If we have separate resolution for each dimension
                    assert ("stateDimensionResolution" in self.configDict and
                            dimName in self.configDict['stateDimensionResolution']), \
                        "State dimension %s has no specified resolution." % dimName
                    self.stateSpace[dimName]["resolution"] \
                            = self.configDict['stateDimensionResolution'][dimName]
                else: # If we have the same value for all dimensions:
                    self.stateSpace[dimName]["resolution"] \
                            = self.configDict['stateDimensionResolution']
        
        
    def setActionSpace(self, actionSpace):
        """ Informs the agent about the action space of the environment
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        if actionSpace.hasContinuousDimensions():
            assert ("actionDimensionResolution" in self.configDict), \
                " Continuous action spaces require that an action dimension resolution be defined."
            discreteActionsPerDimension = self.configDict['actionDimensionResolution']
            self.actionSpace = actionSpace.discretizedActionSpace(discreteActionsPerDimension)
        else:
            self.actionSpace = actionSpace
        
        # Get a list of all actions this agent might take
        self.actions = self.actionSpace.getActionList()
        
        self.agentLog.info("%s got new action-space: %s" % (self.__class__.__name__,
                                                            self.actionSpace))
        
        # Since state and action space are now known, the learner can be initialized
        self._initialize()
    
    def setState(self, state):
        """ Informs the agent of the environment's current state 
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
        super(TDAgent, self).setState(state)
        
    def giveReward(self, reward):
        """ Provides a reward to the agent """
        self.reward += reward
    
    def getAction(self):
        """ Request the next action the agent want to execute """
        # Choose the action, based on the current state
        self.action = self._chooseAction()
        
        # Create an action dictionary 
        # that maps action dimension to chosen action
        actionDictionary = dict()
        for index, actionName in enumerate(self.actionSpace.iterkeys()):
            actionDictionary[actionName] = self.action[index]
        
        # Train the agent
        # NOTE: This has to happen in this method since a method like SARSA
        #       needs to know the action taken in the successor state to 
        #       update the previous state
        self._train(terminalState = False)
        
        # Update all observables
        self._updateObservables()
        
        super(TDAgent, self).getAction()

        return self._generateActionObject(actionDictionary)
   
    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started."""        
        # Train the agent a last time for this episode
        self._train(terminalState = True)
        
        # Update all observables
        self._updateObservables()
        
        # Decay epsilon
        self.epsilon *= self.epsilonDecay    
            
        super(TDAgent, self).nextEpisodeStarted()

    ######################  End COMMAND-HANDLING METHODS ###############################

    def getGreedyPolicy(self):
        """ Returns the optimal greedy policy the agent has found so far """
        return ValueFunctionPolicy(self.tdLearner.functionApproximator,
                                   self.tdLearner.actions)


    def _initialize(self):
        """ Initializes learner as soon as state and action space are known """
        # Create function approximator based on the configuration
        # one now may set a policy in advance, that shall be re-cycled during learning
        # as for transfer learning techniques
        if not hasattr(self, "functionApproximator") or self.functionApproximator == None:
            self.functionApproximator = \
                FunctionApproximator.create(self.configDict["function_approximator"],
                                        self.stateSpace, self.actions)

        # The default feature computation is just to scale the state
        featureFct = lambda state: state
        
        #Create the main TD learner object which is responsible for learning
        self.tdLearner = TD_Learner(self.actions, self.functionApproximator,
                                     featureFct, self.configDict)
    
    def _chooseAction(self):
        """
        Choose action to perform for the given state based on the Q-value of the
        state, action pairs
        """
        # Apply epsilon-greedy action selection
        if random.random() < self.epsilon:
            #Choose random action
            action = random.choice(self.actions)
        else:
            # Compute the action with the optimal q-value 
            action = self.functionApproximator.computeOptimalAction(self.state)

        return action
    
    def _updateObservables(self):
        super(TDAgent, self)._updateObservables()
        
        # Update policy observable
        self.policyObservable.updateFunction(
                lambda state: self.functionApproximator.computeOptimalAction(state)[0])
           
        # Update optimal value function observable
        self.optimalValueFunctionObservable.updateFunction(
                lambda state: self.functionApproximator.computeV(state))
        
        # Update Q-function observable 
        valueAccessFunction = lambda state, action : \
                    self.functionApproximator.computeQ(state, action)
        self.stateActionValuesObservable.updateValues(valueAccessFunction,
                                                      self.actions)

