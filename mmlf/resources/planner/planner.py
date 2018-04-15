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
""" Abstract base class for planners

This module contains the abstract base class "Planner" for planning algorithms,
i.e. algorithms for computing the optimal state-action value
function (and thus the optimal policy) for a given model (i.e. state transition
and expected reward function). Subclasses of Planner must implement the
"plan" method.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from mmlf.resources.function_approximators.tabular_storage \
        import TabularStorage

class PlanningFailedException(Exception): 
    """ An exception indicating that planning has failes """
    pass

class Planner(object):
    """ Abstract base class for planners
    
    This module contains the abstract base class "Planner" for planning 
    algorithms, i.e. algorithms for computing the optimal state-action value
    function (and thus the optimal policy) for a given model (i.e. state 
    transition and expected reward function). Subclasses of Planner must 
    implement the "plan" method.
    
    .. versionadded:: 0.9.9
    """
    
    # A dictionary that will contain a mapping from planner name
    # to the respective planner class
    PLANNER_DICT= None
    
    def __init__(self, stateSpace, functionApproximator, gamma, actions,
                 **kwargs):
        self.stateSpace = stateSpace
        # The function approximator which will store the Q-values
        # Note: This can be a TabularStorage function approximator which
        #       does not generalize at all.
        self.functionApproximator = functionApproximator  
        
        # The discount factor
        self.gamma = gamma
        
        # The available actions
        self.actions = actions
    
    def plan(self, *args, **kwargs):
        raise NotImplementedError("The method plan needs to be implemented in "
                                  "a subclass of Planner")    
    
    @staticmethod
    def getPlannerDict():
        """ Returns dict that contains a mapping from planner name to planner class. """
        # Lazy initialization  
        if Planner.PLANNER_DICT == None:
            from mmlf.resources.planner.trajectory_sampling import TrajectorySamplingPlanner
            from mmlf.resources.planner.prioritized_sweeping import PrioritizedSweepingPlanner
            from mmlf.resources.planner.value_iteration import ValueIterationPlanner
                            
            # A static dictionary containing a mapping from name to FA class 
            Planner.PLANNER_DICT = {'TrajectorySampling': TrajectorySamplingPlanner,
                                    'PrioritizedSweeping': PrioritizedSweepingPlanner,
                                    'ValueIteration': ValueIterationPlanner}
            
        return Planner.PLANNER_DICT
    
    @staticmethod
    def create(plannerSpec, stateSpace, functionApproximator, gamma, actions,
               epsilon=0.0):
        """ Factory method that creates planner based on spec-dictionary. """
        # Determine the planner class
        if plannerSpec['name'] in Planner.getPlannerDict():
            PlannerCls = Planner.getPlannerDict()[plannerSpec['name']]
        else:
            raise Exception("No planner %s known." % plannerSpec['name'])
        
        #Create a planner
        planner = PlannerCls(stateSpace, functionApproximator, gamma, actions,
                             epsilon, **plannerSpec)
        
        return planner
        
    def setStates(self, states):
        """Sets the discrete *states* on which dynamic programming is performed.
        
        Remove Q-Values of state-action pairs that are no longer required.
        NOTE: Setting states is only meaningful for discrete state sets, 
             where the TabularStorage function approximator is used.
        """
        assert isinstance(self.functionApproximator, TabularStorage),\
            "Setting states is only meaningful for discrete state sets, "\
            "where the TabularStorage function approximator is used."
        states = set(states)
        for stateValueStorage in self.functionApproximator.stateValueStorages.values():
            removeItems = []
            for xState in stateValueStorage.values.keys():
                if xState not in states:
                    removeItems.append(xState)
            for item in removeItems:
                stateValueStorage.values.pop(item, None) 
                   
    def _validState(self, checkState):
            """ Checks that checkState is discrete or in the hypercube [0,1]^n """
            return (not checkState.isContinuous()) \
                        or all(0 <= dimValue <= 1 for dimValue in checkState)
         
    def _fullBackup(self, state, action, stateTransitionFct, rewardFct,
                    actions):
        """ Computes the target Q-value for the given state-action pair """
        # Reward is estimated based on the rewardFct
        reward = rewardFct(state, action)
        # Sum over all possible successor states
        target = 0
        sumOfProbabilities = 0.0
        for succState, probability in stateTransitionFct(state, action):
            # Compute the maximal q-value for the successor state (Offpolicy!)
            actionValues = \
                [self.functionApproximator.computeQ(succState, succAction)
                                 for succAction in actions]
            # Add this value weighted with the probability that this successor
            # state occurs to the target 
            target += probability * max(actionValues)
            sumOfProbabilities += probability
        
        # Return reward + gamma*avg(Q(succ))
        if sumOfProbabilities == 0.0:
            return reward
        else:
            return reward + self.gamma * target / sumOfProbabilities
        
    def _sampleBackup(self, reward, succState, actions):
        """ Computes the target Q-value for the given state-action pair """ 
        # Off-Policy
        succStateValue = max(self.functionApproximator.computeQ(succState, 
                                                                succAction)
                                    for succAction in actions)
        return reward + self.gamma * succStateValue
    
    def _selectAction(self, state, actions):
        """ Compute the action that maximizes Q(*state*, action) """
        actionValues =  [(self.functionApproximator.computeQ(state, action), 
                          action) for action in actions]
        return max(actionValues)[1]
