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
# Created: 2009/06/30
""" Agent that performs direct search in the policy space to find a good policy

This agent uses a black-box optimization algorithm to optimize the parameters 
of a parametrized policy such that the accumulated (undiscounted) reward of the
the policy is maximized. 
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import mmlf.framework.protocol
import mmlf.framework.filesystem
from mmlf.agents.agent_base import AgentBase

from mmlf.resources.policy_search.policy_search import PolicySearch
from mmlf.framework.observables import FunctionOverStateSpaceObservable

class DPS_Agent(AgentBase):
    """ Agent that performs direct search in the policy space to find a good policy
    
    This agent uses a black-box optimization algorithm to optimize the parameters 
    of a parametrized policy such that the accumulated (undiscounted) reward of the
    the policy is maximized.
    
    **CONFIG DICT** 
        :policy_search: : The method used for search of an optimal policy in the policy space. Defines policy parametrization and internally used black box optimization algorithm.     
    """
    
    DEFAULT_CONFIG_DICT = {'policy_search' : {'method': 'fixed_parametrization',
                                              'policy': {'type': 'linear', 
                                                         'bias': True, 
                                                         'numOfDuplications': 1},
                                              'optimizer': {'name': 'evolution_strategy',
                                                            'sigma':  1.0,
                                                            'evalsPerIndividual': 10, 
                                                            'populationSize' : 5, 
                                                            'numChildren' :10}}}
    
    def __init__(self, *args, **kwargs):
        
        if self.__class__ == DPS_Agent:
            # Create the agent info
            self.agentInfo = mmlf.framework.protocol.AgentInfo(
                                versionNumber = "0.3",
                                agentName= "Direct Policy Search",
                                continuousState = True,
                                continuousAction = True,
                                discreteAction = True,
                                nonEpisodicCapable = False)
    
        # Calls constructor of base class
        super(DPS_Agent, self).__init__(*args, **kwargs)

        self.accumulatedReward = 0.0
               
        # Prepare logging
        self.policyParameterLog = \
                mmlf.framework.filesystem.LogFile(self.userDirObj,
                                                     'policyParameter', 
                                                     baseRef='policydir')
     
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
    
    def setStateSpace(self, stateSpace):
        """ Informs the agent about the state space of the environment
        
        More information about state spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        super(DPS_Agent, self).setStateSpace(stateSpace) 

        self.stateDims = self.stateSpace.getNumberOfDimensions()
              
        
    def setActionSpace(self, actionSpace):
        """ Informs the agent about the action space of the environment
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        super(DPS_Agent, self).setActionSpace(actionSpace) 
          
        # Lazily initialize policy search method
        self._initialize()
    
    def setState(self, state):
        """ Informs the agent of the environment's current state 
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
        super(DPS_Agent, self).setState(state)
        
    def giveReward(self, reward):
        """ Provides a reward to the agent """
        self.accumulatedReward += reward
    
    def getAction(self):
        """ Request the next action the agent want to execute """
        # Choose the action, based on the current state
        self.action = self.policy.evaluate(self.state)
        
        # Chop action to allowed range
        if self.actionSpace.hasContinuousDimensions():
            self.action = [self.actionSpace.chopContinuousAction(self.action[0])]
        
        # Create an action dictionary 
        # that maps action dimension to chosen action
        actionDictionary = dict()
        for index, actionName in enumerate(self.actionSpace.iterkeys()):
            actionDictionary[actionName] = self.action[index]
        
        super(DPS_Agent, self).getAction()
        
        return self._generateActionObject(actionDictionary)
   
    def nextEpisodeStarted(self, doPolicySearch=True):
        """ Informs the agent that a new episode has started."""
        if doPolicySearch: # Otherwise, the subclass takes care of this alone
            # Store the parameters of the policy used in this episode
            self.policyParameterLog.addText("%s\n" % (self.policy.getParameters()))
    
            # Inform optimizer of achieved performance
            self.policySearch.tellFitness(self.accumulatedReward)
            
            # Get the current policy
            self.policy = self.policySearch.getActivePolicy()
        
        # Update all observables
        self._updateObservables()
        
        # Clean up...
        self.accumulatedReward = 0.0  
        
        super(DPS_Agent, self).nextEpisodeStarted()

    ######################  End COMMAND-HANDLING METHODS ###############################
    
    def getGreedyPolicy(self):
        """ Returns the optimal policy the agent has found so far """
        # Per default. the current policy is assumed to be the best one found
        return self.policySearch.getBestPolicy()
    
    def _initialize(self):
        """ Lazy initialization of the agent once state and action space are known """
        # Lazy initialization of the policy search method
        self.policySearch = PolicySearch.create(self.configDict['policy_search'],
                                                self.stateSpace, 
                                                self.actionSpace, 
                                                self.stateDims)
        
        # Get the current policy
        self.policy = self.policySearch.getActivePolicy() 
        
        # NOTE: Observables can not be created before it is known whether action
        #       space is continuous...
        # An observable that can be used to monitor the agent's optimal policy
        self.optimalPolicyObservable = \
            FunctionOverStateSpaceObservable(title='%s (optimal policy)' % self.__class__.__name__,
                                             discreteValues=not self.actionSpace.hasContinuousDimensions())
        
        # An observable that can be used to monitor the agent's current policy    
        self.currentPolicyObservable = \
            FunctionOverStateSpaceObservable(title='%s (current policy)' % self.__class__.__name__,
                                             discreteValues=not self.actionSpace.hasContinuousDimensions())
    
    def _updateObservables(self):
        super(DPS_Agent, self)._updateObservables()
        
        # Update policy observables
        self.currentPolicyObservable.updateFunction(
                lambda state: self.policy.evaluate(state))
        
        optimalPolicy = self.getGreedyPolicy()
        self.optimalPolicyObservable.updateFunction(
                lambda state: optimalPolicy.evaluate(state))
        

AgentClass = DPS_Agent
AgentName = "Direct Policy Search"

