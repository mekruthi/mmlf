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

# Author: Mark Edgington
# Created: 2007/07/25
""" Module for MMLF interface for agents

This module contains the AgentBase class that specifies the interface that all
MMLF agents have to implement. 
"""

__author__ = "Mark Edgington"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Jan Hendrik Metzen']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import logging
import cPickle

import mmlf.framework.protocol

class PolicyNotStorable(Exception):
    pass

class AgentBase(object):
    """ MMLF interface for agents
    
    Each agent that should be used in the MMLF needs to be derived from this
    class and implements the following methods:
    
    **Interface Methods** 
    
        :setStateSpace: : Informs the agent of the environment's state space  
        :setActionSpace: : Informs the agent of the environment's action space
        :setState: : Informs the agent of the environment's current state
        :giveReward: : Provides a reward to the agent
        :getAction: : Request the next action the agent want to execute
        :nextEpisodeStarted: : Informs the agent that the current episode 
                               has terminated and a new one has started.
    """
    
    def __init__(self, config, baseUserDir, *args, **kwargs):        
        assert self.agentInfo != None, \
               "Agent %s does not set agentInfo!" % self.__class__.__name__
        
        # Create agent logger
        self.agentLog = logging.getLogger('AgentLog')  
                    
        # Dictionary which contains all configuration options specific to this agent.
        # It is VERY important to put ALL configuration options which uniquely determine
        # the behavior of the agent in this dictionary.
        self.configDict = dict()
        if "configDict" in config:
            for key, value in config["configDict"].iteritems():
                if isinstance(value, basestring) \
                      and not isinstance(self.DEFAULT_CONFIG_DICT.get(key, None),
                                         basestring):
                    try:
                        self.configDict[key] = eval(value)
                    except NameError:
                        self.agentLog.info(
                                "Value %s for key %s of agent config could not "
                                "be evaluated. This is not necessarily critical if "
                                "the key ought to be a string." % (value, key))
                        self.configDict[key] = value
                else:
                    self.configDict[key] = value
        
        self.stateSpace = None
        self.actionSpace = None
        
        # Remember the last SARSA tuple
        self.lastState = None
        self.lastAction = None
        self.reward = 0
        self.state = None
        self.action = None
        
        # General attributes
        self.stepCounter = 0
        self.episodeCounter = 0          
        
        # Prepare logging   
        self.userDirObj = baseUserDir
        self.userDirObj.createPath(['%s' % self.__class__.__name__],
                                   refName='agentlogs', baseRef='currentlogdir',
                                   force=True)
    
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
    
    def setStateSpace(self, stateSpace):
        """ Informs the agent about the state space of the environment
        
        More information about state spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        self.stateSpace = stateSpace
        
        self.agentLog.info("%s got new state-space: %s" % (self.__class__.__name__,
                                                           self.stateSpace))
        
    def setActionSpace(self, actionSpace):
        """ Informs the agent about the action space of the environment
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        self.actionSpace = actionSpace
                        
        self.agentLog.info("%s got new action-space: %s" % (self.__class__.__name__,
                                                            self.actionSpace))
    
    def setState(self, state, normalizeState=True):
        """ Informs the agent of the environment's current state.
        
        If *normalizeState* is True, each state dimension is scaled to the
        value range(0,1). More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
        self.state = self.stateSpace.parseStateDict(state)
        
        if normalizeState:
            self.state.scale(0, 1)
        
        self.stepCounter += 1
        
    def giveReward(self, reward):
        """ Provides a reward to the agent """
        pass

    def getAction(self):
        """ Request the next action the agent want to execute """
        # Update the agent's internal SARSA tuple
        self.lastState, self.lastAction, self.reward, self.state, self.action \
                        = self.state, self.action, 0, None, None
    
    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started."""
        # Delete the SARSA tuple
        self.lastState, self.lastAction, self.reward, self.state, self.action = \
                None, None, 0, None, None    
        
        self.stepCounter = 0
        self.episodeCounter += 1

    ########################  END COMMAND-HANDLING METHODS ###############################

    def getGreedyPolicy(self):
        """ Returns the optimal, greedy policy the agent has found so far """
        raise PolicyNotStorable("The agent does not have a method for "
                                "determining greedy policy!")
    
    def storePolicy(self, filePath, optimal=True):
        """ Stores the agent's policy in the given file by pickling it.
        
        Pickles the agent's policy and stores it in the file *filePath*.
        If the agent is based on value functions, a value function policy 
        wrapper is used to obtain a policy object which is then stored.
        
        If optimal==True, the agent stores the best policy it has found so far,
        if optimal==False, the agent stores its current (exploitation) policy.  
        """
        if optimal:
            policy = self.getGreedyPolicy()
        else:
            policy = self.policy

        file = open(filePath, 'w')
        try:
            cPickle.dump(policy, file)
        except TypeError, e:
            import warnings
            warnings.warn("The policy could not be stored. "
                          "Exception: %s" % e) 
        file.close()
        
        return filePath

    def _generateActionObject(self, actionDict):
        """ Creates an action object based on action specified in actionDict """
        return mmlf.framework.protocol.ActionTaken(action=actionDict)
    
    def _updateObservables(self):
        pass
    
