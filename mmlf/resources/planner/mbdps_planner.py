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
# Created: 2010/07/14
""" A planner module which is used in the MBDPS agent. """

import numpy
import logging

from mmlf.framework.state import State
from mmlf.resources.model.model import ModelNotInitialized    
from mmlf.resources.planner.planner import Planner
    
def boundState(state):
    """ Return the given state with each dimension bounded to [-0.1, 1.1] """
    assert type(state) == State
    dimensions = state.dimensions
    # change to numpy array since where operator does not work on States
    state = numpy.array(state) # Create a copy
    state[numpy.where(state < 0.0)] = 0.0
    state[numpy.where(state > 1.0)] = 1.0
    # Change back to State
    state = State(state, dimensions)

    return state

def estimatePolicyOutcome(startState, model, policy, steps, gamma=1.0,
                          returnTrajectory=False):
    """ Estimate a policy's reward and (optional) trajectory based on a model.
    
    This method simulates one episode starting in *startState*, selecting 
    actions based on the given *policy*, samples state transitions based on the
    *model* for a maximum of *steps* steps. It returns the infinite horizion
    discounted reward with discount factor *gamma*. If *returnTrajectory* is 
    true, the state trajectory is returned along with reward.
    """
    state = startState
    trajectory = [startState]
    
    counter = 0
    accumulatedReward = 0.0
    discount = 1.0 
    while counter < steps and not model.isTerminalState(state):
        # Choose the action, based on the current state
        action = policy.evaluate(state)

        # Let the model predict the successor state 
        # and the obtained reward
        reward = model.getExpectedReward(state, action)
        
        # Accumulate the obtained rewards (discounted)
        accumulatedReward += discount*reward
        discount *= gamma
        
        try:
            succState = model.sampleSuccessorState(state, action)
            state = boundState(succState)
        except ModelNotInitialized:
            # Our model is not initialized, we have to break
            break
            
        trajectory.append(state)
        counter += 1

    if not returnTrajectory:
        return accumulatedReward
    else:
        return accumulatedReward, trajectory
    
class MBDPSPlanner(Planner):
    """A planner module which is used in the MBDPS agent. 

    The planner uses a policy search method to search in the space of policies.
    Policies are evaluated based on the return the achieve in trajectory sampled
    from a supplied model. A policy's fitness is set to the average return it
    obtains in several episodes starting from potentially different start states.
    
    Parameters:
        * *fitnessFunction* The fitness function used to evaluate the policy.
                            Defaults to the module's function *estimatePolicyOutcome*.
    
    .. versionadded:: 0.9.9
    
    **CONFIG DICT**
        :gamma: : The discounting factor.
        :planningEpisodes: : The number of simulated episodes that can be conducted before the planning is stopped.
        :evalsPerIndividual: : The number episodes each policy is evaluated.    
    """
    
    DEFAULT_CONFIG_DICT = {'gamma' : 1.0,
                           'planningEpisodes' : 200,
                           'evalsPerIndividual': 1}
    
    def __init__(self, gamma, planningEpisodes, evalsPerIndividual, 
                 fitnessFunction = estimatePolicyOutcome, **kwargs):
        self.gamma = gamma
        self.planningEpisodes = planningEpisodes
        self.evalsPerIndividual = evalsPerIndividual
        self.fitnessFunction = fitnessFunction
            
    
    def plan(self, currentPolicy, policySearch, model, maxEpisodeLength,
             startState=None):
        """ Search for a policy that maximizes the given fitness function.
        
        Search for *self.planningEpisodes* episodes for a policy that maximizes 
        the fitness function *self.fitnessFunction* based on the state transition
        represented by the given *mode*l starting from the given *startState* or
        some start states drawn randomly from the model and a maximum episode
        length of *maxEpisodeLength*. It uses *policySearch* to search for
        a policy that maximizes the fitness function.
        
        At the end of the planning, it is compared whether the optimized policy
        performs better than the current policy the agent follows 
        (*currentPolicy*). If yes, this new policy is returned, otherwise
        *currentPolicy* is returned.  The method returns a three-tuple:
         * The policy that should be followed
         * If planning failed or succeded
         * Whether the returned policy is not *currentPolicy*
        """
        # Determine *evalsPerIndividual* start states for planning
        startStates = []
        if startState is not None:
            # use the actual start state
            startStates.append(startState)
        else:
            # Sample self.evalsPerIndividual start states
            for i in range(self.evalsPerIndividual):                      
                startStates.append(model.drawStartState())
        startStates = list(set(startStates)) # Remove duplicate start states
        
        # Determine if there is an upper bound on the achievable fitness
        maxFitness = model.RMax if hasattr(model, "RMax") else numpy.inf
        
        virtualEpisodeCounter = 0
        try:
            while virtualEpisodeCounter < self.planningEpisodes - 2*self.evalsPerIndividual:
                # Update the policies parameters
                policy = policySearch.getActivePolicy()
                         
                # Estimate the policy's fitness
                policyFitness = \
                    self._estimatePolicyFitness(startStates, model, policy, 
                                                maxEpisodeLength)
                    
                # Inform optimizer of achieved performance
                policySearch.tellFitness(policyFitness)          
                
                virtualEpisodeCounter += self.evalsPerIndividual
                
                # If we have found a policy that achieves the maximal possible 
                # fitness
                if policyFitness == maxFitness:
                    logging.getLogger('').info("Changing policy after planning. "
                                                "Estimate: %s " % maxFitness)
                    return policy, False, True
        except ModelNotInitialized, e:
            logging.getLogger('').info("Planning interrupted after %s steps "
                                        "since model was not able to predict."
                                        % virtualEpisodeCounter)
            
        # If planning was interrupted too early, we continue with old policy
        if virtualEpisodeCounter == 0:
            return currentPolicy, True, False

        # Determine whether we use the best policy found during planning or
        # keep the old policy
        oldPoliciesFitness = \
            self._estimatePolicyFitness(startStates, model, currentPolicy, 
                                        maxEpisodeLength)
        newPoliciesFitness = \
            self._estimatePolicyFitness(startStates, model, 
                                        policySearch.getBestPolicy(), 
                                        maxEpisodeLength)
            
        if newPoliciesFitness > oldPoliciesFitness:
            # Planning has found a better policy 
            logging.getLogger('').info("Changing policy after planning. "
                                       "Estimate: %s " % newPoliciesFitness)
                                                      
            return policySearch.getBestPolicy(), False, True
        else:
            # Planning has not found a policy better than current policy,
            # keep old one
            logging.getLogger('').info("Keeping old policy. Estimate: %s "
                                        % oldPoliciesFitness)
            return currentPolicy, False, False
        
    def _estimatePolicyFitness(self, startStates, model, policy, 
                               maxEpisodeLength):
        """ Estimate the policy's fitness
        
        Estimate the policy's fitness by averaging over self.evalsPerIndividual
        episodes performed by the given *policy* starting in the *startStates*
        and sampling state transitions from *model*. The episode length
        is restricted to a maximal length of *maxEpisodeLength*.
        
        """
        # Get self.evalsPerIndividual fitness samples for policy
        fitnessPredictions = []
        while len(fitnessPredictions) < self.evalsPerIndividual:
            for startState in startStates:
                # Perform one episode with the policy in the 
                # model starting at start state
                fitnessSample = \
                    self.fitnessFunction(startState, model, policy,
                                         maxEpisodeLength, self.gamma) 
                fitnessPredictions.append(fitnessSample)
                if len(fitnessPredictions) >= self.evalsPerIndividual:
                    break
        # Return average fitness
        return float(sum(fitnessPredictions)) / self.evalsPerIndividual
        
    
    