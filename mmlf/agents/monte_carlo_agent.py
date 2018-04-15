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
# Created: 2008/05/16
""" Monte-Carlo learning agent

This module defines an agent which uses Monte Carlo policy evaluation
to optimize its behavior in a given environment
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
import copy
from collections import defaultdict

import mmlf.framework.protocol
from mmlf.agents.agent_base import AgentBase
from mmlf.framework.observables import StateActionValuesObservable

class MonteCarloAgent(AgentBase):
    """ Agent that learns based on monte-carlo samples of the Q-function

    An agent which uses Monte Carlo policy evaluation to optimize its behavior
    in a given environment.
    
    **CONFIG DICT** 
        :gamma: : The discount factor for computing the return given the rewards
        :epsilon: : Exploration rate. The probability that an action is chosen non-greedily, i.e. uniformly random among all available actions
        :visit: : Whether first ("first") or every visit ("every") is used in Monte-Carlo updates
        :defaultQ: : The initially assumed Q-value for each state-action pair. Allows to control initial exploration due to optimistic initialization 
    """

    DEFAULT_CONFIG_DICT = {'gamma' : 1.0,
                           'epsilon' : 0.1,
                           'visit' : "first",
                           'defaultQ' : 0.0}
    
    def __init__(self, *args, **kwargs):
        
        self.agentInfo = mmlf.framework.protocol.AgentInfo(
                            versionNumber="0.3",
                            agentName="Monte Carlo",
                            continuousState = False,
                            continuousAction = False,
                            discreteAction = True,
                            nonEpisodicCapable = False)
    
        # Calls constructor of base class
        super(MonteCarloAgent, self).__init__(*args, **kwargs)
        
        self.hasContinuousActionSpace = False
        
        self.episode = { 'states': [], 'actions': [], 'rewards': []}
        
        self.samples = defaultdict(lambda : 0)
        self.qvalues = defaultdict(lambda: self.configDict['defaultQ']) 
        
        # An observable that can be used to monitor an agents Q-Function
        self.stateActionValuesObservable = \
            StateActionValuesObservable(title='%s Q Function' % self.__class__.__name__)
    
      
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
            
    def setActionSpace(self, actionSpace):
        """ Informs the agent about the action space of the environment
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        ##TODO: Extend to more than one action dimension?
        if len(actionSpace.keys()) > 1:
            raise UserWarning("Error: Currently, only one action dimension is "
                              "possible in Monte-Carlo!")
        
        super(MonteCarloAgent, self).setActionSpace(actionSpace)

        self.actions = copy.copy(self.actionSpace["action"]["dimensionValues"])
    
    def setState(self, state):
        """ Informs the agent of the environment's current state 
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """       
        super(MonteCarloAgent, self).setState(state)
            
        self.episode['states'].append(self.state)
        
    def giveReward(self, reward):
        """ Provides a reward to the agent """
        self.episode['rewards'].append(reward)
        
    
    def getAction(self):
        """ Request the next action the agent want to execute """
        #Choose the action, based on the current state
        actions = copy.copy(self.actions)
        random.shuffle(actions)

        exploration = random.random() < self.configDict['epsilon']
        if exploration:
            maxAction = random.choice(actions)
        else:
            maxValue, foo, maxAction = max((self.qvalues[(self.state, action)], 
                                            random.random(), # break ties randomly!
                                            action) for action in actions)
                
        self.episode['actions'].append(maxAction)
        
        actionDictionary = {'action' : maxAction}
        
        super(MonteCarloAgent, self).getAction()
        
        return self._generateActionObject(actionDictionary)
        
    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started."""
        
        visited_state_actions = set()
                
        trajectory = zip(self.episode['states'], self.episode['actions'])
        
        for step, state_action in enumerate(trajectory):
            if self.configDict['visit'] == "first" and state_action in visited_state_actions:
                continue 

            visited_state_actions.add(state_action)
            
            discReturn = sum(map(lambda x: self.configDict['gamma']**x[0] * x[1],
                                 enumerate(self.episode['rewards'][step:])))
            
            self.qvalues[state_action] = float(self.qvalues[state_action] * self.samples[state_action] + discReturn) \
                                                / (self.samples[state_action] + 1)
                
            self.samples[state_action] = self.samples[state_action] + 1
            
        self.episode = { 'states': [], 'actions': [], 'rewards': []}
        
        # Update Q-function observable
        def valueAccessFunction(state, action):
            if isinstance(action, tuple):
                action = action[0]
            return self.qvalues[(state, action)]
        self.stateActionValuesObservable.updateValues(valueAccessFunction,
                                                      self.actions)
        
        super(MonteCarloAgent, self).nextEpisodeStarted()
        

AgentClass = MonteCarloAgent
AgentName = "Monte-Carlo"
