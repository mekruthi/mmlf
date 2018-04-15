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
""" Planning based on prioritized sweeping. """

import math
import heapq
import logging

from mmlf.resources.model.model import ModelNotInitialized    
from mmlf.resources.planner.planner import Planner, PlanningFailedException
    
class PrioritizedSweepingPlanner(Planner):
    """ Planning based on prioritized sweeping
    
    A planner based on the prioritized sweeping algorithm that
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
        :minSweepDelta: : The minimal TD error that is applied during prioritized sweeping. If no change larger than minSweepDelta remains, the sweep is stopped.
        :updatesPerStep: : The maximal number of updates that can be performed in one sweep.
                     
    """

    
    DEFAULT_CONFIG_DICT = {'updatesPerStep' : 100,
                           'minSweepDelta' : 0.1}
    
    def __init__(self, stateSpace, functionApproximator, gamma, actions, epsilon,
                 minSweepDelta, updatesPerStep, **kwargs):
        super(PrioritizedSweepingPlanner, self).__init__(stateSpace,
                                                         functionApproximator, 
                                                         gamma, actions, **kwargs)
        
        self.minSweepDelta = minSweepDelta
        self.updatesPerStep = updatesPerStep
        
    def plan(self, state, action, stateTransitionFct, 
             invStateTransitionFct, rewardFct, *args, **kwargs):
        """ Computes the optimal value function using prioritized sweeping.

        Computes the optimal state-action value function based on the given
        model consisting of the state transition distribution 
        *stateTransitionFct* and the expected reward function *rewardFct*.
        It starts with the Q-value of the given *state*-*action* pair. 
        The change of the Q-value is propagated back to all predecessor states
        of *state* using the predecessor distribution *invStateTransitionFct*.
        If the expected change of a predecessor state is above self.minSweepDelta,
        this state is queued for updating. The state with the maximal expected
        change in its Q-value is updated next and its change is also 
        propagated back to its predecessor states. When the queue of all 
        states becomes empty or when more than self.updatesPerStep have been 
        conducted, the sweep is stopped. The learned Q-function is used as
        initialization for the next sweep.
        """
        # We use a min heap to implement the priority queue used in
        # prioritized sweeping
        heap = []
        openPairs = dict() # All states that are already in the heap and their delta
        closedPairs = set() # All states that have already been updated
        
        # Compute the change of the Q value of the state-action pair
        try:
            target = self._fullBackup(state, action, stateTransitionFct, 
                                      rewardFct, self.actions)
        except ModelNotInitialized: # Model not properly initialized
            raise PlanningFailedException("Planning failed since model is not "
                                          "initialized")
        
        delta = math.fabs(target - self.functionApproximator.computeQ(state,
                                                                      action))
                
        # If the change is above a threshold
        if delta > self.minSweepDelta:
            # We add the state, action pair to the priority queue with
            # priority -delta (Min heap!)
            heapq.heappush(heap, (-delta, (state, action)))
            openPairs[(state, action)] = -delta
                    
        # We perform self.updatesPerStep updates of state-action pairs 
        for i in range(self.updatesPerStep):
            # If the heap is empty, we stop prioritized sweeping
            if len(heap) == 0:
                break
            
            # Get the next state action pair that should be update
            delta, (state, action) = heapq.heappop(heap)
            openPairs.pop((state, action), None)
            closedPairs.add((state, action))
            logging.getLogger('').debug("i:%s delta: %s heapsize:%s, openPairs: %s" 
                                          % (i, -delta, len(heap), len(openPairs)))
                        
            # Compute target (for the first iteration, use the externally 
            # provided reward)
            if i != 0:
                target = self._fullBackup(state, action, stateTransitionFct, 
                                          rewardFct, self.actions) 
            
            # Do the actual update of the Q value for the state action pair
            self.functionApproximator.train({(state, action) : target})

            # For all actions
            for predAction in self.actions:
                # Determine the predecessor states of state under predAction
                try: 
                    for predState, probability in invStateTransitionFct(state,
                                                                        predAction):
                        # States that are already in the queue are ignored
                        if (predState, predAction) in openPairs:
                            continue                        
                        
                        # States that have already been updated
                        if (predState, predAction) in closedPairs:
                            continue
                        
                        # Check if the state is valid
                        if not self._validState(predState):
                            continue
                        
                        # Compute the estimated change of this state
                        target = self._fullBackup(predState, predAction,
                                                  stateTransitionFct, rewardFct,
                                                  self.actions) 
                        current = self.functionApproximator.computeQ(predState, 
                                                                     predAction)
                        delta = math.fabs(target - current)
                        
                        logging.getLogger('').debug("current: %s target: %s delta:%s" 
                                                        % (current, target, delta))
                        
                        # If the change is above a threshold
                        if delta > self.minSweepDelta:
                            # We add the predecessor state, action pair to the priority 
                            # queue with priority -delta (Min heap!)      
                            heapq.heappush(heap, (-delta, (predState, predAction)))
                            openPairs[(predState, predAction)] = -delta
                            
                except ModelNotInitialized: # Model not properly initialized
                    continue
                
        