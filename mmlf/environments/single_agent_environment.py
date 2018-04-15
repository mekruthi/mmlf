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

""" Abstract base classes for environments in the MMLF.

This module defines abstract base classes from which environments in the MMLF
must be derived.

The following environment base classes are defined:
 :SingleAgentEnvironment: : Base class for environments in which a single agent \
                            can act.

"""

__author__ = "Mark Edgington"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Jan Hendrik Metzen']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import logging

import mmlf.framework.protocol
from mmlf.framework.observables import FloatStreamObservable
from mmlf.framework.observables import TrajectoryObservable

class ImproperAgentException(Exception):
    """ Exception thrown if an improper agent is added to a world. """
    pass

class SingleAgentEnvironment(object):
    """ MMLF interface for environments with a simgle agent
    
    Each environment that should be used in the MMLF needs to be derived from 
    this class and implement the following methods:
    
    **Interface Methods** 
    
        :getInitialState: Returns the initial state of the environment
        :getStateSpace: Returns the state space of the environment
        :getActionSpace: Returns the action space of the environment
        :evaluateAction(actionObject): Evaluate the action defined in *actionObject*
    """
    def __init__(self, config, baseUserDir, useGUI):
        assert self.environmentInfo != None, \
               "Environment %s does not set environmentInfo!" % self.__class__.__name__
        
        self.config = config # Remember for usage in monitor
        
        # Create environment logger
        self.environmentLog = logging.getLogger('%s' % self.__class__.__name__)
        
        # Dictionary which contains all configuration options specific to this environment.
        # It is VERY important to put ALL configuration options which uniquely determine
        # the behavior of the environment in this dictionary.
        self.configDict = {}
        if config != None and "configDict" in config:
            # Try to evaluate all strings
            for key, value in config["configDict"].iteritems():
                if isinstance(value, basestring) \
                   and not isinstance(self.DEFAULT_CONFIG_DICT.get(key, None), 
                                      basestring):
                    try:
                        self.configDict[key] = eval(value)
                    except NameError:
                        self.environmentLog.info(
                            "Value %s for key %s of environment config "
                            "could not be evaluated. This is not necessarily "
                            "critical if the key ought to be a string." % (value, key))
                        self.configDict[key] = value
                else:
                    self.configDict[key] = value
                    
        self.wishQueue = [] # simple list which we use to queue messages to the ISvr

        # some other agent-implementation-specific attributes
        self.agentHasActionSpace = False
        self.agentHasStateSpace = False
        self.agentHasState = False

        # A counter for the number of steps that have been performed in this episode
        self.stepCounter = 0
        # Counter for the episode
        self.episodeCounter = 0
        # Accumulator for the reward gathered in an episode
        self.accumulatedReward = 0.0
                                  
        # Create an observable for the episodes' return and one for the episode 
        # length
        self.returnObservable = \
            FloatStreamObservable(title='%s Episode Return' % self.__class__.__name__,
                                  time_dimension_name='Episode',
                                  value_name='Episode Return')
        self.episodeLengthObservable = \
            FloatStreamObservable(title='%s Episode Length' % self.__class__.__name__,
                                  time_dimension_name='Episode',
                                  value_name='Episode Length')
            
        # An observable that can be used to monitor an agents trajectory through
        # the state action space
        self.trajectoryObservable = \
            TrajectoryObservable(title='%s Trajectory' % self.__class__.__name__)
                
    
    ######################  BEGIN MESSAGE-HANDLING METHODS ###############################
    
    def giveAgentInfo(self, agentInfo):
        """ Check whether an agent is compatible with this environment.
        
        The parameter *agentInfo* contains informations whether an agent is 
        suited for continuous state and/or action spaces, episodic domains
        etc. It is checked whether the agent has the correct capabilties for
        this environment. If not, an 
        :class:`~mmlf.environments.single_agent_environment.py.ImproperAgentException`
        exception is thrown.     

        **Parameters**
        
           :agentInfo: : A dictionary-like object of type
                         :class:`~mmlf.framework.protocol.GiveAgentInfo` 
                         that contains information regarding the agent's
                         capabilities.         
        """
        # Check whether the given agent fulfills all requirements of this environment
        if self.environmentInfo["continuousStateSpace"] and not agentInfo["continuousState"]:
            raise ImproperAgentException("%s agent can not deal with continuous state spaces!" 
                                % agentInfo["agentName"])
        if self.environmentInfo["continuousActionSpace"] and not agentInfo["continuousAction"]:
            raise ImproperAgentException("%s agent can not deal with continuous action spaces!" 
                                % agentInfo["agentName"])
        if self.environmentInfo["discreteActionSpace"] and not agentInfo["discreteAction"]:
            raise ImproperAgentException("%s agent can not deal with discrete action spaces!" 
                                % agentInfo["agentName"])
        if not self.environmentInfo["episodic"] and not agentInfo["nonEpisodicCapable"]:
            raise ImproperAgentException("%s agent can not act in non episodic tasks!" 
                                % agentInfo["agentName"])  
   
    def actionTaken(self, action):
        """ Executes an action chosen by the agent. 
        
        Causes a state transition of the environment based on the specific action
        chosen by the agent. Depending on the successor state, the agent is 
        rewarded, informed about the end of an episodes and/or provided
        with the next state. 
    
        **Parameters**
        
           :action: : A dictionary that specifies for each dimension of the 
                      action space the value the agent has chosen for the 
                      dimension. 
        
        """    
        # Evaluate the given action
        responseDict = self.evaluateAction(actionObject=action)
        reward = responseDict["reward"]
        terminalState = responseDict["terminalState"]
        nextState = responseDict["nextState"]
        startNewEpisode = responseDict["startNewEpisode"]
        
        # reward the agent if a reward has been specified
        if reward != None:
            giveReward = mmlf.framework.protocol.GiveReward(reward=reward)
            self.wishQueue.append(giveReward)
        
        # if the current episode has ended:
        if startNewEpisode == True:
            if terminalState is not None:
                # Inform agent about then terminal state of the finished episode
                terminalStateMsg = mmlf.framework.protocol.SetState(state=terminalState)
                self.wishQueue.append(terminalStateMsg)
            # Start next episode
            startNextEpisode = mmlf.framework.protocol.NextEpisodeStarted()
            self.wishQueue.append(startNextEpisode) 
        
        # send the agent the next state
        giveNextState = mmlf.framework.protocol.SetState(state=nextState)
        self.wishQueue.append(giveNextState)
    
    def getWish(self):
        """ Query the next command object for the interaction server."""        
        # the first times we are polled, we send the state and action space to the agent
        if not self.agentHasStateSpace:
            self.agentHasStateSpace = True
            stateSpace = self.getStateSpace()
            return mmlf.framework.protocol.SetStateSpace(stateSpace=stateSpace)
        
        if not self.agentHasActionSpace:
            self.agentHasActionSpace = True
            actionSpace = self.getActionSpace()
            return mmlf.framework.protocol.SetActionSpace(actionSpace=actionSpace)
        
        if not self.agentHasState:
            self.agentHasState = True
            initialState = self.getInitialState()
            
            # send the initial state
            return mmlf.framework.protocol.SetState(state=initialState)
        
        # if there are any wishes in the wish-queue, then send one of them
        if len(self.wishQueue) > 0:
            wishObject = self.wishQueue.pop(0) # pop off left-most element
            return wishObject
        
        # otherwise, request an action from the agent (indicating to the agent if we're in extended-test mode)
        cmdObject = mmlf.framework.protocol.GetAction()
        return cmdObject    
    ####################  END MESSAGE-HANDLING METHODS  ####################
    
    ###################  BEGIN ENVIRONMENT-SPECIFIC PART ###################
    
    def stop(self):
        """Method which is called when the environment should be stopped """
        self.agentHasActionSpace = False
        self.agentHasStateSpace = False
        self.agentHasState = False

        self.stepCounter = 0
        self.episodeCounter = 0
    
    def getInitialState(self):
        """Returns the initial state of the environment
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
        raise NotImplementedError("This method must be implemented by the "
                                  "subclasses of SingleAgentEnvironment")
    
    def getStateSpace(self):
        """Returns the state space of this environment.
        
        More information about state spaces can be found in
        :ref:`state_and_action_spaces`
        """
        return self.stateSpace
    
    def getActionSpace(self):
        """Return the action space of this environment.
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        return self.actionSpace
 
    def evaluateAction(self, actionObject):
        """ Execute an agent's action in the environment.
        
        Take an actionObject containing the action of an agent, and evaluate 
        this action, calculating the next state, and the reward the agent 
        should receive for having taken this action.
        
        Additionally, decide whether the episode should continue,
        or end after the reward has been  issued to the agent.
        
        This method returns a dictionary with the following keys:
           :rewardValue: : An integer or float representing the agent's reward.
                           If rewardValue == None, then no reward is given to the agent.
           :startNewEpisode: : True if the agent's action has caused an episode
                               to get finished.
           :nextState: : A State object which contains the state the environment
                         takes on after executing the action. This might be the
                         initial state of the next episode if a new episode
                         has just started (startNewEpisode == True)
           :terminalState: : A State object which contains the terminal state 
                             of the environment in the last episode if a new 
                             episode has just started (startNewEpisode == True). 
                             Otherwise None.        
        """
        raise NotImplementedError("This method must be implemented by the "
                                  "subclasses of SingleAgentEnvironment")
    
    ###################  END ENVIRONMENT-SPECIFIC PART ###################
    
    def plotStateSpaceStructure(self, axis):
        """ Plot structure of state space into given axis. 
        
        Just a helper function for viewers and graphic logging.
        """
        return # Do nothing, needs to be implemented in subclass


EnvironmentClass = SingleAgentEnvironment

