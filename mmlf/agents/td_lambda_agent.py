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
""" Agent based on temporal difference learning with eligibility traces
 
This module defines an agent that uses temporal difference learning  (e.g. Sarsa)
with eligibility traces and function approximation
(e.g. linear tile coding CMAC) to optimize its behavior in a given environment
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import mmlf.framework.protocol
from mmlf.agents.td_agent import TDAgent

from mmlf.resources.learning_algorithms.eligibility_traces import EligibilityTrace
from mmlf.framework.observables import StateActionValuesObservable

class TDLambdaAgent(TDAgent):
    """ Agent that implements TD(lambda) RL
    
    An agent that uses temporal difference learning  (e.g. Sarsa)
    with eligibility traces and function approximation (e.g. linear tile
    coding CMAC) to optimize its behavior in a given environment

    **CONFIG DICT** 
        :update_rule: : Whether the learning is on-policy or off-policy.. Can be either "SARSA" (on-policy) or "WatkinsQ" (off-policy)
        :gamma: : The discount factor for computing the return given the rewards
        :epsilon: : Exploration rate. The probability that an action is chosen non-greedily, i.e. uniformly random among all available actions
        :epsilonDecay: : Decay factor for the exploration rate. The exploration rate is multiplied with this value after each episode. 
        :lambda: : The eligibility trace decay rate
        :minTraceValue: : The minimum value of an entry in a trace that is considered to be relevant. If the eligibility falls  below this value, it is set to 0 and the entry is thus no longer updated
        :replacingTraces: : Whether replacing or accumulating traces are used.
        :stateDimensionResolution: : The default "resolution" the agent uses for every state dimension. Can be either an int (same resolution for each dimension) or a dict mapping dimension name to its resolution.
        :actionDimensionResolution: : Per default, the agent discretizes a continuous action space in this number of discrete actions
        :function_approximator: : The function approximator used for representing the Q value function     
     """
    
    DEFAULT_CONFIG_DICT = {'update_rule' : "SARSA",
                           'gamma' : 0.9,
                           'epsilon' : 0.1,
                           'epsilonDecay' : 1.0,
                           'lambda' : 0.9,
                           'minTraceValue' : 0.1,
                           'replacingTraces' : True,
                           'stateDimensionResolution' : 5,
                           'actionDimensionResolution' : 7,
                           'function_approximator' :  {'name' : 'TabularStorage',
                                                       'learning_rate' : 1.0,
                                                       'default' : 0.0}}
    
    def __init__(self, *args, **kwargs):
        
        if self.__class__ == TDLambdaAgent:
            # Create the agent info
            self.agentInfo = mmlf.framework.protocol.AgentInfo(
                                versionNumber = "0.3",
                                agentName = "TD(lambda)",
                                continuousState = True,
                                continuousAction = True,
                                discreteAction = True,
                                nonEpisodicCapable = True)
        
        # Call the constructor of the super class
        super(TDLambdaAgent, self).__init__(*args, **kwargs)
        
        if not 'update_rule' in self.configDict:
            self.configDict['update_rule'] = "SARSA"        
        elif self.configDict['update_rule'] == "WatkinsQ" and self.configDict['lambda'] > 0.0:
            self.agentLog.warning("Using the update rule 'WatkinsQ' with lambda>0. "
                                  "MMLF does neither use WatkinsQ(lambda) nor "
                                  "PengQ(lambda) but the 'naive' Q(lambda) " 
                                  "(compare Sutton/Barto, chapter 7.6)")
        
        #Create Eligibility Traces for all possible actions
        minTraceValue = self.configDict['minTraceValue']
        self.eligibilityTrace = EligibilityTrace(minTraceValue = minTraceValue)
                    
        # An observable that can be used to monitor the agent's eligibility traces
        self.eligibilityTraceObservable = \
            StateActionValuesObservable(title='%s (eligibility trace)' 
                                                    % self.__class__.__name__)
            
            
    ######################  BEGIN COMMAND-HANDLING METHODS #####################
    
    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started."""
        newEpisodeResponse = super(TDLambdaAgent, self).nextEpisodeStarted()

        # Before we continue, we have to reset the eligibility traces
        self.eligibilityTrace.traces.clear()
                    
        return newEpisodeResponse
    
    ######################  End COMMAND-HANDLING METHODS ###############################
      
    def _train(self, terminalState = False):
        """ Train agent on last experience and eligibility traces.
         
        Train the agent using the last (s,a,r,s',a') tuple and the stored
        eligibility traces 
        """
        # Since we are going to update the Q-table for the 
        # pair of lastState, lastAction, we have to check if they are 
        # not None       
        if self.lastState != None and self.lastAction != None:
            # We set the eligibility for the lastState, lastAction pair
            self.eligibilityTrace.setEligibility(self.lastState, 
                                                 self.lastAction, 
                                                 1)
            if terminalState: 
                # If we have reached a terminal state,
                # the target is simply the obtained reward obtained after executing
                # last Action in lastState
                target = self.reward
            else:
                # Otherwise we let the learner compute the delta
                target = self.tdLearner.computeTarget(self.lastState,
                                                      self.lastAction,
                                                      self.reward, self.state,
                                                      self.action)
            
            # Train the learner using the eligibility traces
            traces = self.eligibilityTrace.getTraces()
            self.tdLearner.trainOnTraces(self.lastState, self.lastAction,
                                         target, traces)
        
        # Update eligibility trace observable 
        valueAccessFunction = lambda state, action : \
                    self.eligibilityTrace.getEligibility(state, action)
        self.eligibilityTraceObservable.updateValues(valueAccessFunction,
                                                     self.actions)
        
        # Decay the eligibility by the factor lambda * gamma
        self.eligibilityTrace.decayAllEligibilities(self.configDict['lambda'] 
                                                    * self.configDict['gamma'])
        
    

AgentClass = TDLambdaAgent
AgentName = "Temporal Difference + Eligibility"
