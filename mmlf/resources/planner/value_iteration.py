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
# Created: 2009/08/11
""" Planning based on value iteration

This module contains a planner based on the value iteration algorithm that
allows to compute the optimal state-action value
function (and thus the optimal policy) for a given distribution model (i.e. 
state transition and expected reward function). It is assumed that the MDP is 
finite and that the available actions are defined explicitly.
"""

import logging
import numpy

from mmlf.resources.function_approximators.tabular_storage \
        import TabularStorage
from mmlf.resources.model.model import ModelNotInitialized    
from mmlf.resources.planner.planner import Planner, PlanningFailedException

class ValueIterationPlanner(Planner):
    """ Planning based on value iteration
    
    This module contains a planner based on the value iteration algorithm that
    allows to compute the optimal state-action value
    function (and thus the optimal policy) for a given distribution model (i.e. 
    state transition and expected reward function). It is assumed that the MDP is 
    finite and that the available actions are defined explicitly.
    
    The following parameters must be passed to the constructor:
      * *stateSpace* The state space of the agent (must be finite)
      * *functionApproximator* The function approximator which handles storing
                               the Q-Function
      * *gamma* The discount factor of the MDP
      * *actions* The actions available to the agent
      
    .. versionadded:: 0.9.9
      
    **CONFIG DICT**
        :minimalBellmanError: : The minimal bellman error (sum of TD errors over all state-action pairs) that enforces another iteration. If the bellman error falls below this threshold, value iteration is stopped.      
        :maxIterations: : The maximum number of iterations before value iteration is stopped.
    """
    
    DEFAULT_CONFIG_DICT = {'minimalBellmanError' : 0.1,
                           'maxIterations' : 100}
    
    def __init__(self, stateSpace, functionApproximator, gamma, actions, epsilon,
                 minimalBellmanError, maxIterations, **kwargs):
        super(ValueIterationPlanner, self).__init__(stateSpace,
                                                    functionApproximator, 
                                                    gamma, actions, **kwargs)
        
        self.states = stateSpace.getStateList()
        
        self.minimalBellmanError = minimalBellmanError
        self.maxIterations = maxIterations
        
    def plan(self, stateTransitionFct, rewardFct, isTerminalState,
             *args, **kwargs):
        """ Computes the optimal value function  using value iteration
        
        Computes the optimal state-action value function based on the given
        model consisting of the state transition distribution 
        *stateTransitionFct* , the expected reward function *rewardFct* and
        a method that determines whether a state is a terminal one.
        It computes the q-Value of any combination of states in self.stateSpace
        and actions in self.actions. The iteration is stopped when the bellman error
        in one iteration falls below self.minimalBellmanError or when 
        self.maxIterations are exceeded.
        
        Note: If this method is called more than once (for instance when the
        model has changed), the Q-Function is initialized with the optimal
        Q-function found the last time.
        """
        assert isinstance(self.functionApproximator, TabularStorage),\
                "Value iteration can only be used for discete MDPs with the "\
                "TabularStorage function approximator."        
        iteration = 0
        bellmannResidual = numpy.inf
        while iteration < self.maxIterations and bellmannResidual > self.minimalBellmanError:
            bellmannResidual = 0
            terms = 0
            # For each combination of state and action
            for action in self.actions:
                for state in self.states:
                    if isTerminalState(state):
                        target = 0
                    else:                                                    
                        # Compute the target value for this state-action pair
                        # based on the expected reward and
                        # the Q vales of the successor states.
                        try:
                            target = self._fullBackup(state, action, stateTransitionFct,
                                                      rewardFct, self.actions)
                        except ModelNotInitialized: # Model not properly initialized
                            raise PlanningFailedException("Planning failed since "
                                                          "model is not initialized")
                        
                        bellmannResidual +=\
                             (self.functionApproximator.computeQ(state, action)
                                             - target) ** 2
                    terms += 1

                    self.functionApproximator.train({(state, action) : target})                       
            
            bellmannResidual = bellmannResidual / terms
            logging.getLogger('').info("Iteration %s - Bellman Residual: %s " 
                                            % (iteration, bellmannResidual))
            
            iteration += 1


