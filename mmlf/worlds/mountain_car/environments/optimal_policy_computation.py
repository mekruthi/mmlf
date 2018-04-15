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

import copy
import cPickle
import math
import numpy
import pylab

if __name__ == "__main__":
    import os 
    import sys
    sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
    os.environ["MMLF_RW_PATH"] = "/home/jmetzen/.mmlf/"

def getDiscreteStateSpace(N):
    """ Return discrete state space for mountain car with N values per dimension. """
    from mmlf.framework.spaces import StateSpace
    stateSpaceDict =  {"position": ("discrete", numpy.linspace(-1.2, 0.6 ,N)),
                       "velocity": ("discrete", numpy.linspace(-0.07, 0.07, N))}
    stateSpace = StateSpace()
    stateSpace.addOldStyleSpace(stateSpaceDict, limitType="soft")
    
    return stateSpace

def scaleState(state, stateSpace):
    """ Scale state such that its values fall into the range [0,1]"""
    state = copy.deepcopy(state)
    state.dimensions = map(lambda name: stateSpace[name], 
                           sorted(stateSpace.keys()))                           
    state.scale(0, 1)
    return state


def computeOptimalPolicy(config):
    """ Compute the optimal policy for mountain car for the given *config*. """
    from mmlf.resources.planner.value_iteration import ValueIterationPlanner
    from mmlf.resources.function_approximators.tabular_storage import TabularStorage
    from mmlf.resources.policies.value_function_policy import ValueFunctionPolicy
    from mmlf.worlds.mountain_car.environments.mcar_env \
                                import MountainCarEnvironment
       
    # Discretize state space
    env = MountainCarEnvironment(config=config)
    trueStateSpace = env.stateSpace
    N = env.configDict["N"]
    env.stateSpace = getDiscreteStateSpace(env.configDict["N"])
    
    # Modify stateTransitionFct such that successor states are on the grid
    def stateTransitionFct(state, action):
        for succState, prob in env.stateTransitionFct(state, action):
            protoState = copy.deepcopy(succState)
            protoState.dimensions = copy.deepcopy(succState.dimensions)
            # Get nearest neighbor grid cells in each direction           
            posStepSize = 1.8 / (N - 1)
            velStepSize = 0.14 / (N - 1) 
            for gridCell in env.stateSpace.getStateList():
                if abs(gridCell[0] - succState[0]) < posStepSize \
                    and abs(gridCell[1] - succState[1]) < velStepSize:
                    prob = (1.0 - abs(succState[0] - gridCell[0])/posStepSize) \
                            * (1.0 -abs(succState[1] - gridCell[1])/velStepSize)
                    if prob <= 0.01:
                        continue
                    
                    protoState[0] = gridCell[0]
                    protoState[1] = gridCell[1]
                                
                    yield (protoState, prob)
    
    # Storage structure for Q-table
    functionApproximator = TabularStorage(actions=env.actions, default=0,
                                          learning_rate=1.0) 
        
    # Use value iteration to derive an optimal policy
    planner = ValueIterationPlanner(stateSpace=env.stateSpace,
                                    functionApproximator=functionApproximator,
                                    gamma=0.99, actions=env.actions,
                                    epsilon=0.0, minimalBellmanError=10**-5,
                                    maxIterations=100)
    planner.plan(stateTransitionFct = stateTransitionFct,
                 rewardFct = lambda state, action: env.rewardFct(state, action),
                 isTerminalState = lambda state: env.isTerminalState(state))
    
    ### Store policy ###
    # Convert value function to scaled states
    scaledFunctionApproximator = TabularStorage(actions=env.actions, default=0,
                                                learning_rate=1.0)
    for (state, action), value in functionApproximator.values.iteritems():
        state = scaleState(state, trueStateSpace)
        scaledFunctionApproximator.values[(state, action)] = value

    policy = ValueFunctionPolicy(scaledFunctionApproximator, env.actions)
    
    plotPolicy(env, functionApproximator, scaledFunctionApproximator)
    
    return policy

def plotPolicy(env, functionApproximator, scaledFunctionApproximator):    
    # Plot policy 
    from collections import defaultdict
    position = defaultdict(list)
    velocity = defaultdict(list)
    for state in env.stateSpace.getStateList():
        if env.isTerminalState(state):
            optAction = "T"
        else:
            optAction = functionApproximator.computeOptimalAction(state)

        position[optAction].append(state["position"])
        velocity[optAction].append(state["velocity"])
        
        
    fig = pylab.figure()
    for action, color in zip(["left", "right", "none", "T"], ['r', 'b', 'g', 'k']):
        if len(position[action]) == 0: continue
        pylab.scatter(position[action], velocity[action], c=color, alpha = 1,
                       label = action,)
    
    pylab.xlabel("position")
    pylab.ylabel("velocity")
    fig.savefig(open("policy.png", "w"))
    
    # Plot value function
    from mmlf.resources.function_approximators.knn import KNNFunctionApproximator
    knn = KNNFunctionApproximator(env.stateSpace, env.actions, k = 4,
                                  b_X = 0.025)
    knn.train(scaledFunctionApproximator.values)
    knn.plot(trajectories = [[]], 
             logFile = "value_function.png")
    

if __name__ == "__main__":
    import mmlf
    mmlf.setupConsoleLogging("debug")                
    
    config = {"configDict" : {'maxStepsPerEpisode' : "500", 
                              "accelerationFactor": "0.001",
                              "maxGoalVelocity": "0.07",
                              "positionNoise": "0.0",
                              "velocityNoise": "0.0",
                              "N": "35"}}
    
    policy = computeOptimalPolicy(config)
    
    file = open("optimalPolicy.pickle", 'w')
    try:
        cPickle.dump(policy, file)
    except TypeError, e:
        import warnings
        warnings.warn("The policy could not be stored. Exception: %s" % e) 
