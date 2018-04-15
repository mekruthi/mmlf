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

# 2007-07-25, Mark Edgington

# This module defines the protocol / commands which must be used for communication between the
# interaction server and the agents/environment.

# TODO: implement check() methods in (some of?) these message classes, which raise an error if the
#       message is missing certain keys or has malformed keys.  Then the executive can, if it 
#       desires, call the check() method on a message it receives to ensure that it contains 
#       legitimate data.  This may in particular be useful for enforcing that state or action spaces
#       are in a valid form when they are being communicated.

from copy import deepcopy

#The current version of the protocol:
version = 0.2

class Message(dict):
    """
    A base class which represents a command or response sent between the interaction server and agents/environment.
    
    """
    def __init__(self, **kwargs):
        pass
    
#####  ISvr -> Env commands #####
class GiveAgentInfo(Message):
    """
    Gives the Environment all information about the agents which the agents report on themselves.
    
    AgentInfoList is a list of AgentInfo() objects.  Each agent must make available such an object
    instance named "agentInfo".  i.e. one can access an agents info by looking at agentObj.agentInfo
    
    """
    def __init__(self, agentInfo):
        self["messageName"] = "giveAgentInfo"
        self["agentInfo"] = agentInfo
    
class GetWish(Message):
    """
    Get the next wish/command of the Environment
    
    The environment's transact method should return a single message object in response to receiving
    this message object via transact().
    
    """
    def __init__(self):
        self["messageName"] = "getWish"

#####  (Agent -> ) ISvr -> Env responses #####
class ActionTaken(Message):
    """
    Communicate to the environment (from the interaction server) the action taken by an agent.
    
    This message must be communicated via Env.transact() right after the 'wish' is requested by
    the environment.  In other words, there should be no other message communicated via transact()
    before this message is communicated.
    
    """
    def __init__(self, action=None):
        self["messageName"] = "actionTaken"
        self["action"] = action

#####  [Env -> ISvr] and [ISvr -> Agent] commands #####
# note that, for now, all of the "one or more agents" commands below are implemented for only one
# agent.  Later, the protocol may be extended for use with more than one agent.

# TODO: add SetActiveAgent() class, to permit setting a default agent that other cmds refer to ?

class SetStateSpace(Message):
    "Set the state space of one or more agents"
    def __init__(self, stateSpace):
        self["messageName"] = "setStateSpace"
        self["stateSpace"] = stateSpace

class SetState(Message):
    "Set the state of one or more agents"
    def __init__(self, state):
        self["messageName"] = "setState"
        self["state"] = deepcopy(state) # make a copy

class SetActionSpace(Message):
    "Set the action space of one or more agents"
    def __init__(self, actionSpace):
        self["messageName"] = "setActionSpace"
        self["actionSpace"] = actionSpace

class GetAction(Message):
    """
    Request (demand) an action from one or more agents.
    
    If this request is issued from the Environment, then it will receive a response via a subsequent
    GiveResponse command.
    
    """
    def __init__(self, extendedTestingIsActive=False):
        self["messageName"] = "getAction"
        
        
class GiveReward(Message):
    "Give a reward to one or more agents"
    def __init__(self, reward):
        self["messageName"] = "giveReward"
        self["reward"] = reward


class NextEpisodeStarted(Message): # cleaner replacement of "goalReached" state dimension
    "Communicate to an agent that a new episode has begun"
    def __init__(self):
        self["messageName"] = "nextEpisodeStarted"

#####  Environment and Agent capabilities/properties classes #####
# these classes must be implemented by agents and environments to report to the ISvr what its
# configuration and/or capabilities are.

class AgentInfo(dict):
    """
    A class which represents the capabilities of an agent.
    """
    def __init__(self, **kwargs):
        "These values must be overridden in by the agent"
        self["policyCacheable"] = False
        self["continuousState"] = True
        self["continuousAction"] = False
        self["discreteAction"] = True
        self["communicationCapable"] = False
        self["nonEpisodicCapable"] = False # can deal with environments in which there are no episodes
        self["versionNumber"] = 0.0
        
        if kwargs != {}: # if there are keyword arguments
            self.update(kwargs) # update the appropriate dictionary entries
        

class EnvironmentInfo(dict):
    """
    A class which represents the configuration of an environment.
    """
    def __init__(self, **kwargs):
        "These values must be overridden in by the environment"
        self["environmentName"] = "nonameEnvironment" #this is the name of the environment
        self["continuousStateSpace"] = True # is any dimension of the state space continuous?
        self["continuousActionSpace"] = False # is any dimension of the action space continuous?
        self["discreteActionSpace"] = True # is any dimension of the action space discrete?
        self["episodic"] = True
        self["stochastic"] = False # is the way the environment responds to actions deterministic or stochastic?
        self["versionNumber"] = 0.0
        self["communicationCapable"] = False # this is set to true if the environment wants to 
                                                                          # read or manipulate the communication of the agents
        
        if kwargs != {}: # if there are keyword arguments
            self.update(kwargs) # update the appropriate dictionary entries





