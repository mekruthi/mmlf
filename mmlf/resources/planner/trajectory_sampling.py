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
""" Planning based on trajectory sampling.

This module contains a planner based on the trajectory sampling algorithm that
allows to compute  an improved state-action value function (and thus policy) 
for a given sample model. The MDP's state space need not be discrete and finite
but it is assumed that there is only a finite number of actions that are 
defined explicitly.
"""

import random

from mmlf.resources.model.model import ModelNotInitialized    
from mmlf.resources.planner.planner import Planner
    
class TrajectorySamplingPlanner(Planner):
    """ Planning based on trajectory sampling.
    
    A planner based on the trajectory sampling algorithm that allows to compute 
    an improved state-action value function (and thus policy) for a given sample
    model. The MDP's state space need not be discrete and finite
    but it is assumed that there is only a finite number of actions that are 
    defined explicitly.
    
    The following parameters must be passed to the constructor:
      * *stateSpace* The state space of the agent
      * *functionApproximator* The function approximator which handles storing
                               the Q-Function
      * *gamma* The discount factor of the MDP
      * *actions* The actions available to the agent
      
    .. versionadded:: 0.9.9
      
    **CONFIG DICT**
        :maxTrajectoryLength: : The maximal length of a trajectory before a new trajectory is started.
        :updatesPerStep: : The maximal number of updates that can be performed in one planning call.
        :onPolicy: : Whether the trajectory is sampled from the on-policy distribution.
    """
    
    DEFAULT_CONFIG_DICT = {'updatesPerStep' : 100,
                           'maxTrajectoryLength' : 50,
                           'onPolicy' : True}
    
    def __init__(self, stateSpace, functionApproximator, gamma, actions, 
                 epsilon, maxTrajectoryLength, updatesPerStep, onPolicy,
                  **kwargs):
        super(TrajectorySamplingPlanner, self).__init__(stateSpace,
                                                        functionApproximator, 
                                                        gamma, actions, **kwargs)
        
        self.epsilon = epsilon
        self.maxTrajectoryLength = maxTrajectoryLength
        self.updatesPerStep = updatesPerStep
        self.onPolicy = onPolicy
        
    def plan(self, state, sampleStartState, sampleSuccessorState, rewardFct,
             isTerminalState, *args, **kwargs):
        """ Computes the optimal value function using trajectory sampling.
        
        This method samples repeatedly trajectories along which the Q-function
        is updated. The start state of the trajectories is sampled from
        *sampleStartState*, the successor state for an action selected
        according to the internal value function is sampled using 
        *sampleSuccessorState*. The reward of this action is determine by 
        *rewardFct*. For the given transition, the Q-Learning sample backup
        target is computed and the internal value function updated accordingly.
        Afterwards, it is checked whether the successor state is a terminal
        state by using *isTerminalState*. If yes or if the trajectory has the 
        length self.maxTrajectoryLength, a new start state is sampled an a new 
        trajectory started. This is repeated, until self.updatesPerStep steps
        have passed.
        """
        # Always start at the current state
        self.trajectoryState = state
        self.trajectoryLength = 0

        for step in range(self.updatesPerStep):
            try:
                # Determine action based on policy
                if not self.onPolicy or random.random() > self.epsilon:
                    # Pick greedy action
                    action = self._selectAction(self.trajectoryState, self.actions)
                else:
                    # Pick random action
                    action = random.choice(self.actions)
                # Sample successor state and reward
                succState = sampleSuccessorState(self.trajectoryState, action)
                reward = rewardFct(self.trajectoryState, action)
                # Compute Q-Learning sample backup target and update function
                # approximator
                if isTerminalState(succState):
                    target = reward
                else:
                    target = self._sampleBackup(reward, succState, self.actions)
                self.functionApproximator.train({(self.trajectoryState, action) : target})
                # Check whether we should start a new episode
                if isTerminalState(succState) or \
                        self.trajectoryLength >= self.maxTrajectoryLength:
                    # Spend the rest of the time to sample trajectories from 
                    # actual start states
                    self.trajectoryState = sampleStartState()
                    self.trajectoryLength = 0
                else:
                    self.trajectoryState = succState
                    self.trajectoryLength += 1
            except ModelNotInitialized: 
                # We cannot follow current trajectory, start at a new state
                self.trajectoryState = sampleStartState()
                
