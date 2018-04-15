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
# Created: 2009/07/20
""" Fitted R-Max agent

Fitted R-Max is a model-based RL algorithm that uses the RMax heuristic for
exploration control, uses a fitted function approximator (even though this can
be configured differently), and uses Dynamic Programming (boosted by prioritized
sweeping) for deriving a value function from the model. Fitted R-Max learns 
usually very sample-efficient (meaning that a good policy is learned with only a 
few interactions with the environment) but requires a huge amount of 
computational resources.
""" 

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
from collections import defaultdict

import mmlf.framework.protocol
from mmlf.framework.observables import FloatStreamObservable
from mmlf.agents.agent_base import AgentBase

from mmlf.resources.model.model import Model, ModelNotInitialized
from mmlf.resources.planner.planner import Planner, PlanningFailedException
from mmlf.resources.policies.value_function_policy import ValueFunctionPolicy
from mmlf.resources.function_approximators.function_approximator \
        import FunctionApproximator
from mmlf.resources.function_approximators.tabular_storage \
        import TabularStorage
from mmlf.framework.observables import StateActionValuesObservable, \
                                        FunctionOverStateSpaceObservable

class FittedRMaxAgent(AgentBase):
    """ Fitted R-Max agent
    
    Fitted R-Max is a model-based RL algorithm that uses the RMax heuristic for
    exploration control, uses a fitted function approximator (even though this can
    be configured differently), and uses Dynamic Programming (boosted by prioritized
    sweeping) for deriving a value function from the model. Fitted R-Max learns 
    usually very sample-efficient (meaning that a good policy is learned with only a 
    few interactions with the environment) but requires a huge amount of 
    computational resources.
    
    .. seealso::
        Nicholas K. Jong and Peter Stone,
        "Model-based function approximation in reinforcement learning",
        in "Proceedings of the 6th International Joint Conference on Autonomous Agents and Multiagent Systems" 
        Honolulu, Hawaii: ACM, 2007, 1-8, http://portal.acm.org/citation.cfm?id=1329125.1329242.
    
    **CONFIG DICT**
        :gamma: : The discount factor for computing the return given the rewards#
        :min_exploration_value: : The agent explores in a state until the given exploration value (approx. number of exploratory actions in proximity of state action pair) is reached for all actions
        :RMax: : An upper bound on the achievable return an agent can obtain in a single episode
        :planner: : The algorithm used for planning, i.e. for optimizing the policy based on a learned model
        :model: : The algorithm used for learning a model of the environment
        :function_approximator: : The function approximator used for representing the Q value function
        :actionDimensionResolution: : Per default, the agent discretizes a continuous action space in this number of discrete actions             
    """
    
    DEFAULT_CONFIG_DICT = {'gamma' : 0.99,
                           'min_exploration_value' : 1.0,
                           'RMax' : 0.0,
                           'planner' : {'name' : "PrioritizedSweeping",
                                        'updatesPerStep' : 1000,
                                        'minSweepDelta' : 0.1},
                           'model' : {'name': 'KNNModel',
                                      'k': 100,
                                      'b_Sa': 0.03,
                                      'exampleSetSize': 2500},
                           'function_approximator' : {'name': 'KNN',
                                                      'k': 20,
                                                      'b_X': 0.01},
                            'actionDimensionResolution' : 9}
    
    def __init__(self, *args, **kwargs):
                
        # Create the agent info
        self.agentInfo = mmlf.framework.protocol.AgentInfo(
                            versionNumber = "0.3",
                            agentName= "Fitted R-Max",
                            continuousState = True,
                            continuousAction = False,
                            discreteAction = True,
                            nonEpisodicCapable = True)

        
        # Calls constructor of base class        
        super(FittedRMaxAgent, self).__init__(*args, **kwargs)
     
        self.functionApproximator = None
            
        self.userDirObj.createPath(['model'], refName='modelDir',
                                   baseRef='agentlogs', force=True)
        
        # An observable that stores the exploration value
        self.explorationValueObservable = \
            FloatStreamObservable(title='%s Exploration Value' % self.__class__.__name__,
                                  time_dimension_name='Step',
                                  value_name='Exploration Value')
                
        # An observable that can be used to monitor an agents greedy policy
        self.policyObservable = \
            FunctionOverStateSpaceObservable(title='%s (greedy policy)'
                                                     % self.__class__.__name__,
                                             discreteValues=True)
        # An observable that can be used to monitor an agents optimal value function
        self.optimalValueFunctionObservable = \
            FunctionOverStateSpaceObservable(title='%s (optimal value function)' 
                                                    % self.__class__.__name__,
                                                 discreteValues=False)
        # An observable that can be used to monitor an agents Q-Function
        self.stateActionValuesObservable = \
            StateActionValuesObservable(title='%s Q Function' 
                                                   % self.__class__.__name__)
        # An observable that can be used to monitor the expected reward
        self.expectedRewardObservable = \
            StateActionValuesObservable(title='%s Reward Expectation' 
                                                   % self.__class__.__name__)     
     
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################
    def setActionSpace(self, actionSpace):
        """ Informs the agent about the action space of the environment
        
        More information about action spaces can be found in 
        :ref:`state_and_action_spaces`
        """
        if actionSpace.hasContinuousDimensions():
            assert ("actionDimensionResolution" in self.configDict), \
                """ Continuous action spaces require that an action dimension resolution be defined. """
            discreteActionsPerDimension = self.configDict['actionDimensionResolution']
            self.actionSpace = actionSpace.discretizedActionSpace(discreteActionsPerDimension)
        else:
            self.actionSpace = actionSpace
        
        # Get a list of all actions this agent might take
        self.actions = self.actionSpace.getActionList()
        
        self.agentLog.info("%s got new action-space: %s" % (self.__class__.__name__,
                                                            self.actionSpace))
        
        # Since state and action space are now known, the learner can be initialized
        self._initialize()
    
    
    def setState(self, state):
        """ Informs the agent of the environment's current state 
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """           
        super(FittedRMaxAgent, self).setState(state)
        
    def giveReward(self, reward):
        """ Provides a reward to the agent """
        self.reward = reward
    
    def getAction(self):
        """ Request the next action the agent want to execute """
        if self.lastState != None:
            # Inform the model about the outcome of the last action
            self.model.addExperience(self.lastState, self.lastAction, self.state,
                                     self.reward)
            # Planning
            self._updatePolicy()  
#            if self.stepCounter % 100 == 0:
#                import cProfile; 
#                cProfile.runctx("self._updatePolicy()", globals(), locals(), 
#                                "profiling_data_%s.dat" % self.stepCounter)  
        else:
            # Inform the model about a new start state 
            self.model.addStartState(self.state)
            
        if self.functionApproximator != None:
            # Compute the action with the optimal q-value 
            self.action = self.functionApproximator.computeOptimalAction(self.state)
        else:
            # Randomly choose an action
            self.action = random.choice(self.actions)
        
        # Create an action dictionary that maps 
        # action dimension to chosen action
        actionDictionary = dict()
        for index, actionName in enumerate(self.actionSpace.iterkeys()):
            actionDictionary[actionName] = self.action[index]
        
        # Update explorationValueObservable
        self.explorationValueObservable.addValue(self.stepCounter,
                                                 self.model.getExplorationValue(self.state,
                                                                                self.action))
        
        super(FittedRMaxAgent, self).getAction()
        
        self.agentLog.debug("Episode %s Step: %s State: %s Action: %s" 
                             % (self.episodeCounter, self.stepCounter,
                                self.lastState, self.lastAction))
        
        return self._generateActionObject(actionDictionary)

    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started."""
        # If the agent has actually reached a terminal state
        if self.state is not None:
            # Inform the model about the outcome of the last action
            self.model.addExperience(self.lastState, self.lastAction, self.state,
                                     self.reward)
            # We have reached a terminal state
            self.model.addTerminalState(self.state)
            # Update policy 
            self._updatePolicy()
                
        super(FittedRMaxAgent, self).nextEpisodeStarted()

    def getGreedyPolicy(self):
        """ Returns the optimal greedy policy the agent has found so far """
        return ValueFunctionPolicy(self.functionApproximator,
                                   self.actions)

    def _initialize(self):
        """ Lazy initialization of the agent once state and action space are known """
        # Data structure to store value function of discrete MDP during planning
        self.tabularStorage = TabularStorage(stateSpace=None,
                                             actions=self.actions,
                                             **self.configDict["function_approximator"])
        
        assert  self.configDict["planner"]['name'] == "PrioritizedSweeping", \
                 "Fitted-RMax works currently only with Prioritized Sweeping " \
                 " planner"
        # The object that performs planning
        self.planner = Planner.create(self.configDict["planner"],
                                      self.stateSpace,
                                      self.tabularStorage,
                                      self.configDict['gamma'], self.actions)
        
        # Choose action model class based on conf
        self.model = Model.create(self.configDict["model"], self, self.stateSpace,
                                  self.actionSpace, self.userDirObj) 
    
        # TODO: Just a hack, should be refactored soon
        if "plotting" in self.configDict: 
            self.configDict["plotting"]["modelRasterPoints"] = \
                [self.configDict["plotting"]["modelRasterPoints"]
                    for i in range(self.stateSpace.getNumberOfDimensions())]
        
    def _updatePolicy(self):
        """ Update policy using prioritized sweeping based on the internal model
        
        Construct a discrete, finite MDP based on the state transitions and 
        reward expectations learned by the models.
        For state-action pairs that have not been explored sufficiently, 
        be optimistic. i.e. assume that these states have a value that is equal 
        to the maximal achievable reward (R-Max).
        """
        
        # The states of the constructed MDP
        states = self.model.getStates()
        states.extend([("s_term",), ("s_opt",)])
        self.planner.setStates(states)

        # Generate the discrete, RMax-optimistic version of the MDP to be 
        # solved. This MDP is derived from the learned model.
        discreteRMaxMDP = DiscreteRMaxMDP(self.model, self.configDict["RMax"],
                                          self.configDict["min_exploration_value"],
                                          self.stateSpace.hasContinuousDimensions())
        
        # Perform prioritized sweeping to compute optimal value function
        # for the MDP specified by the model starting from the MDP state "nnState"
        try:
            nnState = self.model.getNearestNeighbor(self.lastState)
            self.planner.plan(state=nnState, action=self.lastAction,
                              sampleStartState=None,
                              sampleSuccessorState=None, 
                              stateTransitionFct=lambda state, action : discreteRMaxMDP.stateTransitionFct(state, action),
                              invStateTransitionFct=lambda state, action : discreteRMaxMDP.invStateTransitionFct(state, action), 
                              rewardFct=lambda state, action : discreteRMaxMDP.rewardFct(state, action), 
                              isTerminalState=None)
        except (PlanningFailedException, ModelNotInitialized) :
            self.agentLog.info("Planning failed!")

        
        # Get the computed qValues and remove the artificial states "s_term" and
        # "s_opt"
        qValues = self.planner.functionApproximator.getPlainValues()   
        for action in self.actions:
            qValues.pop((("s_term",), action), None)
            qValues.pop((("s_opt",), action), None)
        
        # Create a function approximator that generalizes the compute q-values
        # to a value function over the whole continuous state space.
        self.configDict["function_approximator"]["learning_rate"] = 1.0 
        functionApproximator = \
            FunctionApproximator.create(self.configDict["function_approximator"],
                                        self.stateSpace, self.actions)
        
        # Use the RMax function approximator wrapper to ensure that 
        # underexplored states have value RMax
        self.functionApproximator = \
                 RMaxFunctionApproximatorWrapper(functionApproximator, 
                                                 self.stateSpace,
                                                 self.actions,
                                                 self.model,
                                                 self.configDict["RMax"],
                                                 self.configDict["min_exploration_value"])
        
        self.functionApproximator.train(qValues) 

        # Update observables
        self._updateObservables(discreteRMaxMDP)

        
    def _updateObservables(self, discreteRMaxMDP):
        super(FittedRMaxAgent, self)._updateObservables()
        
        # Update policy observable 
        self.policyObservable.updateFunction(
                lambda state: self.functionApproximator.computeOptimalAction(state)[0])
           
        # Update optimal value function observable
        self.optimalValueFunctionObservable.updateFunction(
                lambda state: self.functionApproximator.computeV(state))
        
        # Update Q-function observable 
        valueAccessFunction = lambda state, action : \
                    self.functionApproximator.computeQ(state, action)
        self.stateActionValuesObservable.updateValues(valueAccessFunction, 
                                                      self.actions)
        
        # Update reward expectation observable 
        valueAccessFunction = lambda state, action : \
                    discreteRMaxMDP.rewardFct(state, action)
        self.expectedRewardObservable.updateValues(valueAccessFunction,
                                                   self.actions)        
        
class DiscreteRMaxMDP(object):
    """ Derives a discrete, RMax-optimistic MDP from a learned model. """
    
    def __init__(self, model, rMax, minExplorationValue, discretize=True):
        self.model = model
        self.rMax = rMax
        self.minExplorationValue = minExplorationValue
        self.discretize = discretize
        
    # The reward function that returns the expected reward when executing a
    # certain action in a certain state
    def rewardFct(self, state, action):
        if state == ("s_opt",) or \
             self.model.getExplorationValue(state, action) < self.minExplorationValue:
            # When in the artificial state "S_opt" assume that maximal
            # achievable reward RMax. This implements optimism in the face 
            # of uncertainty
            return self.rMax
        elif state == ("s_term",) or self.model.isTerminalState(state):
            # All terminal states are consuming and do provide no additional
            # reward
            return 0.0
        else:
            # Return the reward the model predicts for this state action 
            # pair
            return self.model.getExpectedReward(state, action)

    # The state transition function that returns an iterator over all 
    # possible successor states (of state when applying action) along with
    # the probability that these particular state is the successor state.
    def stateTransitionFct(self, state, action):
        if state in [("s_term",), ("s_opt",)] or self.model.isTerminalState(state):
            # All terminal and underexplored states ("s_opt") directly 
            # transition into the artificial consuming terminal state "s_term" 
            yield (("s_term",), 1.0)
        else:
            if self.model.getExplorationValue(state, action) \
                            < self.minExplorationValue:
                # All states that have not been explored sufficiently 
                # directly transition into an artificial state "s_opt"
                # that  
                yield (("s_opt",), 1.0)
            else: 
                # Compute a generator that iterates over all possible 
                # successor states
                
                # A dictionary that collects all possible successors along 
                # with their probability
                succStates = defaultdict(float)
                # Iterate over the successor states predicted by the model
                for succState, prob in self.model.getSuccessorDistribution(state, 
                                                                           action):
                    if self.discretize: # we have to discretize
                        # Find up to 10 states in the discrete MDP (the
                        # 'neighbors') that are most similar to succState
                        nearestNeighbors = self.model.getNearestNeighbors(succState,
                                                                          k=10,
                                                                          b=0.01)
                        # Update probability of states to be the successor state
                        for weight, neighbor in nearestNeighbors:
                            # Add the probability to the sum of probabilities
                            # for this neighbor
                            succStates[neighbor] += prob * weight
                    else: # Problem is already discrete
                        succStates[succState] = prob
                            
                # Iterate over all states of the MDP that might be 
                # successors of state
                for succState, prob in succStates.iteritems():
                    if prob >= 0.01:
                        yield succState, prob

    # The state transition function that returns an iterator over all 
    # possible predecessor states that might transition to *state* when 
    # applying action) along with the probability that these particular 
    # state is the predecessor state.
    # NOTE: This is only required for the backward sweep in prioritized
    #       sweeping
    def invStateTransitionFct(self, state, action):
        if state in [("s_term",), ("s_opt",)]:
            # We assume that the predecessor of s_term is s_term and of 
            # s_opt is s_opt. 
            yield (state, 1.0)
        else:
            # Compute a generator that iterates over all possible 
            # predecessor states
            
            # A dictionary that collects all possible predecessors along 
            # with their probability
            predStates = defaultdict(float)
            # Iterate over the predecessor states predicted by the model
            for predState, prob in self.model.getPredecessorDistribution(state, 
                                                                         action):
                if self.discretize: # we have to discretize
                    # Find up to 10 states in the discrete MDP (the
                    # 'neighbors') that are most similar to predState
                    nearestNeighbors = self.model.getNearestNeighbors(predState,
                                                                      k=10,
                                                                      b=0.01)
    
                    # Update probability of states to be the successor state
                    for weight, neighbor in nearestNeighbors:
                        # Add the probability to the sum of probabilities
                        # for this neighbor
                        predStates[neighbor] += prob * weight
                else: # Problem is already discrete
                    predStates[predState] = prob
        
            # Iterate over all states of the MDP that might be predecessors
            # of state
            for predState, prob in predStates.iteritems():
                if prob >= 0.01:
                    yield predState, prob
    


class RMaxFunctionApproximatorWrapper(FunctionApproximator):
    """ A Wrapper for function approximators for usage in Fitted-R-Max
    
    This wrapper behaves like the wrapped *functionApproximator* instance except
    in cases when the Q-Value of a state-action pair should be computed
    whose exploration value is below *minExplorationValue*. In that case,
    the maximum achievable value *RMax* is returned.
    """
    def __init__(self, functionApproximator, stateSpace, actions, model,
                 RMax, minExplorationValue):
        self.functionApproximator = functionApproximator
        self.stateSpace = stateSpace
        self.actions = actions
               
        self.model = model
        self.RMax = RMax
        self.minExplorationValue = minExplorationValue
                   
    def computeQ(self, state, action):
        """ Computation of Q-values based on optimism in the face of uncertainty  
        
        Return RMax for underexplored *state*-*action* pairs otherwise the the  
        output of the wrapped function approximator.
        """
        if self.model.getExplorationValue(state, action) \
                        < self.minExplorationValue:
            return self.RMax
        else:
            return self.functionApproximator.computeQ(state, action)
    
    def train(self, trainingSet):
        """ Train the wrapped function approximator with *trainingSet* """
        self.functionApproximator.train(trainingSet)


AgentClass = FittedRMaxAgent
AgentName = "Fitted-RMax"
