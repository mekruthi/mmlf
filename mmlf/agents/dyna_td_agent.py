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
# Created: 2009/03/23
# Revised: 2010/03/01
""" The Dyna-TD agent module

This module contains the Dyna-TD agent class.
It uses temporal difference learning along with 
learning a model of the environment and is based on the
Dyna architecture.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import mmlf.framework.protocol
from mmlf.agents.td_lambda_agent import TDLambdaAgent
from mmlf.resources.model.model import Model
from mmlf.resources.planner.planner import Planner, PlanningFailedException

class DynaTDAgent(TDLambdaAgent):
    """ Agent that learns based on the DYNA architecture.
    
    Dyna-TD uses temporal difference learning along with 
    learning a model of the environment and doing planning in it.
    
    **CONFIG DICT** 
        :gamma: : The discount factor for computing the return given the rewards
        :epsilon: : Exploration rate. The probability that an action is chosen non-greedily, i.e. uniformly random among all available actions
        :lambda: : The eligibility trace decay rate
        :minTraceValue: : The minimum value of an entry in a trace that is considered to be relevant. If the eligibility falls  below this value, it is set to 0 and the entry is thus no longer updated
        :update_rule: : Whether the learning is on-policy or off-policy.. Can be either "SARSA" (on-policy) or "WatkinsQ" (off-policy)
        :stateDimensionResolution: : The default "resolution" the agent uses for every state dimension. Can be either an int (same resolution for each dimension) or a dict mapping dimension name to its resolution.
        :actionDimensionResolution: : Per default, the agent discretizes a continuous action space in this number of discrete actions
        :planner: : The algorithm used for planning, i.e. for optimizing the policy based on a learned model
        :model: : The algorithm used for learning a model of the environment
        :function_approximator: : The function approximator used for representing the Q value function     
    """
    DEFAULT_CONFIG_DICT = {'stateDimensionResolution' : 5,
                           'actionDimensionResolution' : 7,
                           'gamma' : 0.99,
                           'epsilon' : 0.1,
                           'lambda' : 0.0,
                           'minTraceValue' : 0.5,
                           'update_rule' : "WatkinsQ",
                           'planner' : {'name' : "TrajectorySampling",
                                        'updatesPerStep' : 100,
                                        'maxTrajectoryLength' : 50,
                                        'onPolicy' : True},
                           'model' :  {'name': 'KNNModel',
                                       'k': 20,
                                       'b_Sa': 0.03,
                                       'exampleSetSize': 2500},
                           'function_approximator' :  {'name' : 'CMAC',
                                                       'learning_rate' : 0.5,
                                                       'update_rule' : 'exaggerator',
                                                       'number_of_tilings' : 10,
                                                       'default' : 0.0}}
    
    def __init__(self, *args, **kwargs):
                
        # Create the agent info
        self.agentInfo = mmlf.framework.protocol.AgentInfo(
                            versionNumber = 0.3,
                            agentName= "Dyna_TD",
                            continuousState = True,
                            continuousAction = True,
                            discreteAction = True,
                            nonEpisodicCapable = True)
        
        super(DynaTDAgent, self).__init__(*args, **kwargs)      
     
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################    
    def getAction(self):
        """
        This method is called by the interaction server to request that the agent select an action
        and return this action.
        """
        # Modelling and planning
        if self.lastState != None:
            # Inform the model about the outcome of the last action
            self.model.addExperience(self.lastState, self.lastAction, self.state,
                                     self.reward)
        
            # Use artificial experience from the model to improve policy
            # Use prioritized sweeping...
            self._planning()
            
            self.agentLog.debug("Episode %s Step: %s" 
                                % (self.episodeCounter, self.stepCounter))
        else:
            # Inform the model about a new start state 
            self.model.addStartState(self.state)  

        # Learning takes place in the super class
        return super(DynaTDAgent, self).getAction()

    def nextEpisodeStarted(self):
        """
        This method is called by the interaction server to indicate that a new episode has begun
        (and the old episode has finished).
        
        This method must return a NewEpisodeResponse message in response.  Through this message
        the agent communicates whether it would like to request an extended-test.
        
        """
        # If the agent has actually reached a terminal state
        if self.state is not None:
            # Inform the model about the outcome of the last action
            self.model.addExperience(self.lastState, self.lastAction, self.state,
                                     self.reward)
            # We have reached a terminal state
            self.model.addTerminalState(self.state)
            # Update policy 
            self._planning()

        # Learning takes place in the super class
        return super(DynaTDAgent, self).nextEpisodeStarted()


    def _initialize(self):
        """ Initializes learner as soon as state and action space are known """
        super(DynaTDAgent, self)._initialize()
    
        # The object that performs prioritized sweeping
        self.planner = Planner.create(self.configDict["planner"],
                                      self.stateSpace,
                                      self.functionApproximator,
                                      self.configDict['gamma'], self.actions,
                                      self.configDict['epsilon'])
        
        # Choose action model class based on conf
        self.model = Model.create(self.configDict["model"], self, self.stateSpace,
                                  self.actionSpace, self.userDirObj) 
    
    def _planning(self):
        """ Update policy using dynamic programming """
        
        # Define functions used within planners
        def sampleStartState():
            return self.model.drawStartState()
        def sampleSuccessorState(state, action):
            return self.model.sampleSuccessorState(state, action)        
        def stateTransitionFct(state, action):
            return self.model.getSuccessorDistribution(state, action)
        def invStateTransitionFct(state, action):
            return self.model.getPredecessorDistribution(state, action)
        def rewardFct(state, action):
            return self.model.getExpectedReward(state, action)
        def isTerminalState(state):
            return self.model.isTerminalState(state)
        
        # Do the actual planning
        try:
            self.planner.plan(state=self.lastState, action=self.lastAction,
                              sampleStartState=sampleStartState,
                              sampleSuccessorState=sampleSuccessorState, 
                              stateTransitionFct=stateTransitionFct,
                              invStateTransitionFct=invStateTransitionFct, 
                              rewardFct=rewardFct, isTerminalState=isTerminalState)
        except PlanningFailedException, e:
            self.agentLog.debug(str(e))
            

AgentClass = DynaTDAgent
AgentName = "Dyna TD"
