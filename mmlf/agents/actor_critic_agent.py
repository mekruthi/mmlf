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
# Created: 2010/05/28

""" Agent that learns based on the actor-critic architecture.

This module contains an agent that learns based on the actor critic
architecture. It uses standard TD(lambda) to learn the value function of
the critic and updates the preferences of the actor based on the TD error. 
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random

import mmlf.framework.protocol

from mmlf.agents.td_lambda_agent import TDLambdaAgent
from mmlf.resources.function_approximators.function_approximator\
     import FunctionApproximator
from mmlf.framework.observables import StateActionValuesObservable

class ActorCriticAgent(TDLambdaAgent):
    """ Agent that learns based on the actor-critic architecture.

    This agent learns based on the actor critic architecture.   
    It uses standard TD(lambda) to learn the value function of
    the critic. For this reason, it subclasses TDLambdaAgent. The main
    difference to TD(lambda) is the means for action selection. Instead of
    deriving an epsilon-greedy policy from its Q-function, it learns an 
    explicit stochastic policy. To this end, it maintains preferences for each
    action in each state. These preferences are updated after each action
    execution according to the following rule:
    
    .. math::

        p(s,a) = p(s,a) + \delta, 
        
    where delta is the TD error

    .. math::
    
       \delta = r + \gamma V(s') - V(s)
    Action selection is based on a Gibbs softmax distribution:

    .. math::
    
       \pi(s,a) = \\frac{exp(\\tau^{-1}p(s,a))}{\sum_{b \in A} exp(\\tau^{-1}p(s,b))}
    where tau is a temperature parameter.
    
    Note that even though preferences are stored in a function approximator 
    such that in principle, action preferences could be generalized over the
    state space, continuous state spaces are not yet supported.
    
    .. versionadded:: 0.9.9
       Added Actor-Critic agent
    
    **CONFIG DICT** 
        :gamma: : The discount factor for computing the return given the rewards
        :lambda: : The eligibility trace decay rate
        :tau: : Temperature parameter used in the Gibbs softmax distribution for action selection
        :minTraceValue: : The minimum value of an entry in a trace that is considered to be relevant. If the eligibility falls  below this value, it is set to 0 and the entry is thus no longer updated
        :update_rule: : Whether the learning is on-policy or off-policy.. Can be either "SARSA" (on-policy) or "WatkinsQ" (off-policy)
        :stateDimensionResolution: : The default "resolution" the agent uses for every state dimension. Can be either an int (same resolution for each dimension) or a dict mapping dimension name to its resolution.
        :actionDimensionResolution: : Per default, the agent discretizes a continuous action space in this number of discrete actions.
        :function_approximator: : The function approximator used for representing the Q value function
        :preferences_approximator: The function approximator used for representing the action preferences (i.e. the policy)
    """
    
    DEFAULT_CONFIG_DICT = {'gamma' : 0.0,
                           'lambda' : 0.9,
                           'tau' : 0.2,
                           'minTraceValue' : 0.5,
                           'stateDimensionResolution' : 5,
                           'actionDimensionResolution' : 7,
                           'update_rule' : "SARSA",
                           'function_approximator' :  {'name' : 'TabularStorage',
                                                       'learning_rate' : 0.1,
                                                       'default' : 0.0},
                           'preferences_approximator' :  {'name' : 'TabularStorage',
                                                          'learning_rate' : 1.0,
                                                          'default' : 0.0}}
    
    def __init__(self, *args, **kwargs):
        # Create the agent info
        self.agentInfo = \
            mmlf.framework.protocol.AgentInfo(# Which communication protocol 
                                                 # version can the agent handle?
                                                 versionNumber = "0.3",
                                                 # Name of the agent (can be 
                                                 # chosen arbitrarily) 
                                                 agentName= "Actor Critic", 
                                                 # Can the agent be used in 
                                                 # environment with continuous
                                                 # state spaces?
                                                 continuousState = False,
                                                 # Can the agent be used in 
                                                 # environment with continuous
                                                 # action spaces?
                                                 continuousAction = False,
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
        super(ActorCriticAgent, self).__init__(*args, **kwargs)
                
        # An observable that can be used to monitor an agents preferences
        self.preferencesObservable = \
            StateActionValuesObservable(title='%s Preferences' % self.__class__.__name__)
    
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
    
    def getAction(self):
        """ Request the next action the agent want to execute """
        # We can now modify the preferences for the action taken in the last step
        if self.lastState is not None:
            self._updatePreferences()
        
        return super(ActorCriticAgent, self).getAction()
                
    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started.""" 
        # We can now modify the preferences for the action taken in the last step
        if self.lastState is not None:
            self._updatePreferences()
        
        return super(ActorCriticAgent, self).nextEpisodeStarted()
    
    ########################  END COMMAND-HANDLING METHODS ###############################
    
    def _initialize(self):
        """ Initializes learner as soon as state and action space are known """
        # The actor learns preferences for actions in certain states  
        # Create function approximator for this preference function 
        # based on the configuration
        self.preferencesApproximator = \
            FunctionApproximator.create(self.configDict["preferences_approximator"],
                                        self.stateSpace, self.actions)
        
        super(ActorCriticAgent, self)._initialize()
    
    def _updatePreferences(self):
        """ Update the preferences of the agent for the last chosen action """
        # Compute the critic (in form of a TD error)
        tdError = self.reward + self.configDict['gamma'] * self.functionApproximator.computeV(self.state)  \
                        - self.functionApproximator.computeV(self.lastState)
        
        # Updates actor's preferences accordingly 
        target = self.preferencesApproximator.computeQ(self.lastState,
                                                       self.lastAction) + tdError 
        trainDict = {(self.lastState, self.lastAction) : target}
        self.preferencesApproximator.train(trainDict) 
        
        # Update Q-function observable 
        valueAccessFunction = lambda state, action : \
                    self.preferencesApproximator.computeQ(state, action)
        self.preferencesObservable.updateValues(valueAccessFunction, self.actions)
        
    
    def _chooseAction(self):
        "Chooses an action from the action space"
        preferences = [self.preferencesApproximator.computeQ(self.state, action)
                            for action in self.actions] 
        probabilityMassFunction = computeProbabilityMasses(preferences, 
                                                           self.configDict['tau'])
        
        randValue = random.random()
        accumulator = 0.0
        for index, probabilityMass in enumerate(probabilityMassFunction):
            accumulator += probabilityMass
            if accumulator >= randValue:
                return self.actions[index]

        # For rare cases where the probability masses do not sum to 1 due to 
        # numerical imprecision
        return self.actions[-1]
        

def computeProbabilityMasses(preferences, tau):
    """ Compute the action probabilities based on their preferences
    
    This function computes the action probabilities in state s as
    pi(s, a_i) = exp(tau**-1 * p(s, a_i)) /sum(exp(tau**-1 p(s,a))) 
    based on the actions' preferences. The function uses internally the 
    decimal package to circumvent floating point overflows/underflows.    
    """
    import decimal

    exponents = map(lambda x: decimal.Decimal(str(1.0/tau*x)), preferences) 
    values = map(lambda x: x.exp(), exponents)
    denominator = sum(values)

    return map(lambda x: float(x/denominator), values)
        
    
    

# Each module that implements an agent must have a module-level attribute 
# "AgentClass" that is set to the class that implements the AgentBase superclass
AgentClass = ActorCriticAgent
# Furthermore, the name of the agent has to be assigned to "AgentName". This 
# name is used in the GUI. 
AgentName = "Actor Critic"
