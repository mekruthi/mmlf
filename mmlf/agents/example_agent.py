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

""" MMLF agent that chooses actions in a round-robin manner.

This agent's sole purpose is to give an example of how to write an agent.
It should not be used for any actual learning.  
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import mmlf.framework.protocol

from mmlf.agents.agent_base import AgentBase

# Each agent has to inherit directly or indirectly from AgentBase
class ExampleAgent(AgentBase):
    """ MMLF agent that chooses actions in a round-robin manner. """

    DEFAULT_CONFIG_DICT = {}
    
    def __init__(self, *args, **kwargs):

        # Create the agent info
        self.agentInfo = \
            mmlf.framework.protocol.AgentInfo(# Which communication protocol 
                                                 # version can the agent handle?
                                                 versionNumber = "0.3",
                                                 # Name of the agent (can be 
                                                 # chosen arbitrarily) 
                                                 agentName= "Round Robin", 
                                                 # Can the agent be used in 
                                                 # environment with contiuous
                                                 # state spaces?
                                                 continuousState = True,
                                                 # Can the agent be used in 
                                                 # environment with continuous
                                                 # action spaces?
                                                 continuousAction = True,
                                                 # Can the agent be used in 
                                                 # environment with discrete
                                                 # action spaces?
                                                 discreteAction = True,
                                                 # Can the agent be used in
                                                 # non-episodic environments
                                                 nonEpisodicCapable = True)
    
        # Calls constructor of base class
        # After this call, the agent has an attribute "self.configDict",
        # that contains the information from config['configDict'].
        # The values of this dict are evaluated, i.e. instead of '100' (string),
        # the key 'Reward log frequency' will have the same value 100 (int).
        super(ExampleAgent, self).__init__(*args, **kwargs)
        
        # The superclass AgentBase implements the methods setStateSpace() and
        # setActionSpace() which set the attributes stateSpace and actionSpace
        # They can be overwritten if the agent has to modify these spaces
        # for some reason
        self.stateSpace = None
        self.actionSpace = None
        
        # The agent keeps track of the sum of all rewards it obtained
        self.rewardValue = 0
    
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
            
    def setActionSpace(self, actionSpace):
        """ Informs the agent about the action space of the environment
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        super(ExampleAgent, self).setActionSpace(actionSpace)
        
        # We can only deal with one-dimensional action spaces
        assert self.actionSpace.getNumberOfDimensions() == 1
        
        # Get a list of all actions this agent might take
        self.actions = self.actionSpace.getActionList()
        # Get name of action dimension
        self.actionDimensionName = self.actionSpace.getDimensionNames()[0]
        # Create an iterator that iterates in a round-robin manner over available actions
        self.nextActionIterator = __import__("itertools").cycle(self.actions)        
        
    def getAction(self):
        """ Request the next action the agent want to execute """
        # Get next action  from iterator
        # We are only interested in the value of the first (and only) dimension,
        # thus the "0"
        nextAction = self.nextActionIterator.next()[0] 
        # Create a dictionary that maps dimension name to chosen action
        actionDictionary = {self.actionDimensionName : nextAction}
        
        # Call super class method since this updates some internal information
        # (self.lastState, self.lastAction, self.reward, self.state, self.action)
        super(ExampleAgent, self).getAction()
        
        # Generate mmlf.framework.protocol.ActionTaken object
        return self._generateActionObject(actionDictionary)
    
# Each module that implements an agent must have a module-level attribute 
# "AgentClass" that is set to the class that implements the AgentBase superclass
AgentClass = ExampleAgent
# Furthermore, the name of the agent has to be assigned to "AgentName". This 
# name is used in the GUI. 
AgentName = "RoundRobin"
