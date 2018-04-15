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
# Created: 2011/04/05
""" A linear markov chain environment. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

from copy import deepcopy

from mmlf.framework.spaces import StateSpace, ActionSpace
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.environments.single_agent_environment import SingleAgentEnvironment

# Each environment has to inherit directly or indirectly from SingleAgentEnvironment
class LinearMarkovChainEnvironment(SingleAgentEnvironment):
    """ A linear markov chain.
    
    The agent starts in the middle of this linear markov chain. He can either
    move right or left. The chain is not stochastic, i.e. when the agent 
    wants to move right, the state is decreased with probability 1 by 1.  
    When the agent wants to move left, the state is increased with probability 1
    by 1 accordingly.
    
    .. versionadded:: 0.9.10
       Added LinearMarkovChain environment
    
    **CONFIG DICT**
        :length: : The number of states of the linear markov chain
    
    """
    
    # Add default configuration for this environment to this static dict
    # This specific parameter controls how long the linear markov chain is
    # (i.e. how many states there are)
    DEFAULT_CONFIG_DICT = {"length" : 21}
    
    def __init__(self, useGUI, *args, **kwargs):
        # Create the environment info
        self.environmentInfo = \
            EnvironmentInfo(# Which communication protocol version can the 
                            # environment handle?
                            versionNumber="0.3",
                            # Name of the environment (can be chosen arbitrarily) 
                            environmentName="LinearMarkovChain",
                            # Is the action space of this environment discrete?
                            discreteActionSpace=True,
                            # Is the environment episodic?
                            episodic=True,
                            # Is the state space of environment continuous?
                            continuousStateSpace=False,
                            # Is the action space of environment continuous?
                            continuousActionSpace=False,
                            # Is the environment stochastic?
                            stochastic=False)

        # Calls constructor of base class
        # After this call, the environment has an attribute "self.configDict",
        # The values of this dict are evaluated, i.e. instead of '100' (string),
        # the key 'length' will have the same value 100 (int).
        super(LinearMarkovChainEnvironment, self).__init__(useGUI=useGUI, *args, **kwargs)
               
        # The state space of the linear markov chain
        oldStyleStateSpace =  {"field": ("discrete", range(self.configDict["length"]))}
        
        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")
        
        # The action space of the linear markov chain
        oldStyleActionSpace =  {"action": ("discrete", ["left", "right"])}
        
        self.actionSpace = ActionSpace()
        self.actionSpace.addOldStyleSpace(oldStyleActionSpace, limitType="soft")
        
        # The initial state of the environment
        self.initialState =  {"field": self.configDict["length"] / 2}
        # The current state is initially set to the initial state
        self.currentState = deepcopy(self.initialState)

    ########################## Interface Functions #####################################
    def getInitialState(self):
        """ Returns the initial state of the environment """
        self.environmentLog.debug("Episode starts in state '%s'." 
                                    % (self.initialState['field']))
        return self.initialState
    
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
        action = actionObject['action']
        previousState = self.currentState['field']
        
        # Change state of environment deterministically
        if action == 'left':
            self.currentState['field'] -= 1
        else:
            self.currentState['field'] += 1
            
        self.environmentLog.debug("Agent chose action '%s' which caused a transition from '%s' to '%s'." 
                                    % (action, previousState, self.currentState['field']))
        
        #Check if the episode is finished (i.e. the goal is reached)
        episodeFinished = self._checkEpisodeFinished()
        
        terminalState = self.currentState if episodeFinished else None
        
        if episodeFinished:
            self.episodeLengthObservable.addValue(self.episodeCounter,
                                                  self.stepCounter + 1)
            self.returnObservable.addValue(self.episodeCounter,
                                           -self.stepCounter)
            self.environmentLog.debug("Terminal state '%s' reached." 
                                            % self.currentState['field'])
            self.environmentLog.info("Episode %s lasted for %s steps." 
                                     % (self.episodeCounter, self.stepCounter  + 1))
            
            reward = 10 if self.currentState['field'] != 0 else -10
            
            self.stepCounter = 0
            self.episodeCounter += 1
            
            # Reset the simulation to the initial state (always the same)
            self.currentState = deepcopy(self.initialState)
        else:
            reward = -1
            self.stepCounter += 1
        
        resultsDict = {"reward" : reward, 
                       "terminalState" : terminalState,
                       "nextState" : self.currentState,
                       "startNewEpisode" : episodeFinished}
        return resultsDict
        
    def _checkEpisodeFinished(self):
        """ Checks whether the episode is finished.
        
        An episode is finished whenever the leftmost or rightmost state of the
        chain is reached.
        """        
        return self.currentState['field'] in [0, self.configDict['length']-1]
    
# Each module that implements an environment must have a module-level attribute 
# "EnvironmentClass" that is set to the class that inherits from SingleAgentEnvironment
EnvironmentClass = LinearMarkovChainEnvironment
# Furthermore, the name of the environment has to be assigned to "EnvironmentName".
# This  name is used in the GUI. 
EnvironmentName = "Linear Markov Chain"

