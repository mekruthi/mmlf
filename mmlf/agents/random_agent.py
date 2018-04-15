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

# Author: Jan Hendrik Metzen  (jhm@informatik.uni-bremen.de)
# Created: 2007/07/23

""" MMLF agent that acts randomly

This module defines a simple agent that can interact with an environment.
It chooses all available actions with the same probability.

This module deals also as an example of how to implement an MMLF agent.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

from collections import defaultdict

import mmlf.framework.protocol

from mmlf.agents.agent_base import AgentBase

# Each agent has to inherit directly or indirectly from AgentBase
class RandomAgent(AgentBase):
    """ Agent that chooses uniformly randomly among the available actions. """
    
    # Add default configuration for this agent to this static dict
    # This specific parameter controls after how many steps we send information 
    # regarding the accumulated reward to the logger.
    DEFAULT_CONFIG_DICT = {'Reward_log_frequency' : 100}
    
    def __init__(self, *args, **kwargs):
        # Create the agent info
        self.agentInfo = \
            mmlf.framework.protocol.AgentInfo(# Which communication protocol 
                                                 # version can the agent handle?
                                                 versionNumber = "0.3",
                                                 # Name of the agent (can be 
                                                 # chosen arbitrarily) 
                                                 agentName= "Random", 
                                                 # Can the agent be used in 
                                                 # environments with continuous
                                                 # state spaces?
                                                 continuousState = True,
                                                 # Can the agent be used in 
                                                 # environments with continuous
                                                 # action spaces?
                                                 continuousAction = True,
                                                 # Can the agent be used in 
                                                 # environments with discrete
                                                 # action spaces?
                                                 discreteAction = True,
                                                 # Can the agent be used in
                                                 # non-episodic environments
                                                 nonEpisodicCapable = True)
    
        # Calls constructor of base class
        # After this call, the agent has an attribute "self.configDict",
        # The values of this dict are evaluated, i.e. instead of '100' (string),
        # the key 'Reward log frequency' will have the same value 100 (int).
        super(RandomAgent, self).__init__(*args, **kwargs)
        
        # The superclass AgentBase implements the methods setStateSpace() and
        # setActionSpace() which set the attributes stateSpace and actionSpace
        # They can be overwritten if the agent has to modify these spaces
        # for some reason
        self.stateSpace = None
        self.actionSpace = None
        
        # The agent keeps track of all rewards it obtained in an episode
        # The rewardDict implements a mapping from the episode index to a list
        # of all rewards it obtained in this episode
        self.rewardDict = defaultdict(list)

    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
    
    def setStateSpace(self, stateSpace):
        """ Informs the agent about the state space of the environment
        
        More information about state spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        # We delegate to the superclass, which does the following:
        # self.stateSpace = stateSpace
        # We need not implement this method for this, but it is given in order
        # to show what is going on...
        super(RandomAgent, self).setStateSpace(stateSpace) 

        
    def setActionSpace(self, actionSpace):
        """ Informs the agent about the action space of the environment
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        # We delegate to the superclass, which does the following:
        # self.actionSpace = actionSpace
        # We need not implement this method for this, but it is given in order
        # to show what is going on...
        super(RandomAgent, self).setActionSpace(actionSpace) 
    
    def setState(self, state):
        """ Informs the agent of the environment's current state 
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
        # We delegate to the superclass, which does the following:
        #     self.state = self.stateSpace.parseStateDict(state) # Parse state dict
        #     self.state.scale(0, 1) # Scale state such that each dimension falls into the bin (0,1)
        #     self.stepCounter += 1 # Count how many steps have passed
        
        # We need not implement this method for this, but it is given in order
        # to show what is going on...
        super(RandomAgent, self).setState(state)

        
    def getAction(self):
        """ Request the next action the agent want to execute """
        # Each action of the agent corresponds to one step
        action = self._chooseRandomAction()
        
        # Call super class method since this updates some internal information
        # (self.lastState, self.lastAction, self.reward, self.state, self.action)
        super(RandomAgent, self).getAction()
        
        return action
    
    def giveReward(self, reward):
        """ Provides a reward to the agent """
        self.rewardDict[self.episodeCounter].append(reward) # remember reward
        # Send message about the accumulated reward every 
        # self.configDict['Reward log frequency'] episodes to logger 
        if self.stepCounter % self.configDict['Reward_log_frequency'] == 0:
            self.agentLog.info("Reward accumulated after %s steps in episode %s: %s" 
                                % (self.stepCounter, self.episodeCounter,
                                   sum(self.rewardDict[self.episodeCounter])))
        
    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started."""
        # We delegate to the superclass, which does the following:
        #     self.episodeCounter += 1
        #     self.stepCounter = 0
        super(RandomAgent, self).nextEpisodeStarted()
    
    ########################  END COMMAND-HANDLING METHODS ###############################
    
    def _chooseRandomAction(self):
        "Chooses an action randomly from the action space"
        
        assert self.actionSpace, "Error: Action requested before actionSpace "\
                                 "was specified"
        
        # We sample a random action from the action space
        # This returns a dictionary with a mapping from action dimension name
        # to the sample value.
        # For instance: {"gasPedalForce": "extreme", "steeringWheelAngle": 30}
        actionDictionary = self.actionSpace.sampleRandomAction()

        # The action dictionary has to be converted into an
        # mmlf.framework.protocol.ActionTaken object.
        # This is done using the _generateActionObject method
        # of the superclass
        return self._generateActionObject(actionDictionary)

# Each module that implements an agent must have a module-level attribute 
# "AgentClass" that is set to the class that inherits from Agentbase
AgentClass = RandomAgent
# Furthermore, the name of the agent has to be assigned to "AgentName". This 
# name is used in the GUI. 
AgentName = "Random"
