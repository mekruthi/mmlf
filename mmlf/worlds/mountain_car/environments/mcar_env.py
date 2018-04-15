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

""" Module that implements the mountain car environment and its dynamics. """
# Refactored: 2010-01-0f by Jan Hendrik Metzen

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
from math import cos

from mmlf.framework.spaces import StateSpace, ActionSpace
from mmlf.framework.state import State
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.environments.single_agent_environment import SingleAgentEnvironment

class MountainCarEnvironment(SingleAgentEnvironment):
    """ The mountain car environment. 
    
    In the mountain car environment, the agent has to control a car which is
    situated somewhere in a valley between two hills. The goal of the agent
    is to reach the top of the right hill. Unfortunately, the engine of the
    car is not strong enough to reach the top of the hill directly from many
    start states. Thus, it has first to drive in the wrong direction to gather
    enough potential energy. 
    
    The agent can either accelerate left, right, or coast. Thus, the action
    space is discrete with three discrete actions. The agent observes
    two continuous state components: The current position and velocity of the
    car. The start state of the car is stochastically initialised. 
    
    **CONFIG DICT**
        :maxStepsPerEpisode: : The number of steps the agent has maximally to reach the goal. Benchmark default is "500".
        :accelerationFactor: : A factor that influences how strong the cars engine is relative to the slope of the hill. Benchmark default is "0.001".
        :maxGoalVelocity: : Maximum velocity the agent might have when reaching the goal. If smaller than 0.07, this effectively makes the task MountainPark instead of MountainCar. Benchmark default is "0.07" 
        :positionNoise: : Noise that is added to the agent's observation of the position. Benchmark default is "0.0"
        :velocityNoise: : Noise that is added to the agent's observation of the velocity. Benchmark default is "0.0"

    """
    
    DEFAULT_CONFIG_DICT = {'maxStepsPerEpisode' : 500,    
                           'accelerationFactor' : 0.001,
                           'maxGoalVelocity' : 0.07,
                           'positionNoise' : 0.0,
                           'velocityNoise' : 0.0}
    
    def __init__(self, config, useGUI, *args, **kwargs):
        
        self.environmentInfo = EnvironmentInfo(versionNumber="0.3",
                                               environmentName="Mountain Car",
                                               discreteActionSpace=True,
                                               episodic=True,
                                               continuousStateSpace=True,
                                               continuousActionSpace=False,
                                               stochastic=True)
        
        # Add value for N to config dict (required for discretization
        # in optimal policy computation)
        if "N" not in config["configDict"]:
            config["configDict"]["N"] = "50"
        
        super(MountainCarEnvironment, self).__init__(config, useGUI=useGUI, *args, **kwargs) 
        
        # configuration
        self.randomStarts = True
        
        # Some constants
        self.minPosition = -1.2  # Minimum car position
        self.maxPosition = 0.6   # Maximum car position (past goal)
        self.maxVelocity = 0.07  # Maximum velocity of car
        self.goalPosition = 0.5  # Goal position - how to tell we are done
        
        # If "maxGoalVelocity" is not set in configDict, set it to maximal 
        # velocity
        if not "maxGoalVelocity" in self.configDict:
            self.configDict["maxGoalVelocity"] = self.maxVelocity
        
        # The current state of the system
        self.state = None

        # Some counters
        self.overallStepCounter = 0
                
        # State and action space definition
        oldStyleStateSpace =   {"position": ("continuous", [(self.minPosition,
                                                             self.maxPosition)]), 
                                "velocity": ("continuous", [(-self.maxVelocity,
                                                             self.maxVelocity)])}

        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")

        self.actions = ["left", "right", "none"]
        oldStyleActionSpace =  {"thrust": ("discrete", self.actions)}

        self.actionSpace = ActionSpace()
        self.actionSpace.addOldStyleSpace(oldStyleActionSpace, limitType="hard")
        
        if useGUI:
            from mmlf.gui.viewers import VIEWERS
            from mmlf.gui.viewers.trajectory_viewer import TrajectoryViewer
            from mmlf.worlds.mountain_car.environments.mcar_policy_viewer \
                    import MountainCarPolicyViewer
            from mmlf.worlds.mountain_car.environments.mcar_valuefunction_viewer \
                    import MountainCarValueFunctionViewer
            # Add general trajectory viewer
            VIEWERS.addViewer(lambda : TrajectoryViewer(self.stateSpace), 
                              'TrajectoryViewer')
            VIEWERS.addViewer(lambda : MountainCarPolicyViewer(self.stateSpace),
                              'MountainCar PolicyViewer')
            VIEWERS.addViewer(lambda : MountainCarValueFunctionViewer(self.stateSpace),
                              'MountainCar ValueFunctionViewer')
    
    ########################## Interface Functions #####################################        
    def getInitialState(self):
        """ Returns the initial state of the environment
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
        if self.randomStarts: # random start state
            def randomInInterval(min, max):
                "Returns a random number between min and max"
                return min + (random.random() * (max - min))
            
            position = randomInInterval(self.minPosition, self.goalPosition)
            velocity = randomInInterval(-self.maxVelocity, self.maxVelocity)
        else: # deterministically start in (-0.5, 0.0)
            position = -0.5
            velocity = 0.0
        self.state = {"position": position, "velocity": velocity}
        return self._stateForAgent(self.state)
    
    def evaluateAction(self, actionObject):
        """ Execute an agent's action in the environment.
        
        Take an actionObject containing the action of an agent, and evaluate 
        this action, calculating the next state, and the reward the agent 
        should receive for having taken this action.
        
        Additionally, decide whether the episode should continue,
        or end after the reward has been  issued to the agent.
        
        This method returns a dictionary with the following keys:
           :rewardValue: : An integer or float representing the agent's reward.
                           If rewardValue == None, then no reward is given to the agent.
           :startNewEpisode: : True if the agent's action has caused an episode
                               to get finished.
           :nextState: : A State object which contains the state the environment
                         takes on after executing the action. This might be the
                         initial state of the next episode if a new episode
                         has just started (startNewEpisode == True)
           :terminalState: : A State object which contains the terminal state 
                             of the environment in the last episode if a new 
                             episode has just started (startNewEpisode == True). 
                             Otherwise None.        
        """
        # Remember state before executing action
        previousState = self.state
        
        # Execute the action which was chosen by the agent
        self.state, prob = list(self.stateTransitionFct(self.state, 
                                                        actionObject['thrust']))[0]
        self.stepCounter += 1
        
        #Check if the episode is finished (i.e. the goal is reached)
        episodeFinished = False
        terminalState = None
        
        if self.isTerminalState(self.state):
            episodeFinished = True
            terminalState = self._stateForAgent(self.state)
            self.environmentLog.info("Episode %s: Goal reached after %s steps." 
                                     % (self.episodeCounter, self.stepCounter))
        elif self.stepCounter >= self.configDict["maxStepsPerEpisode"]:
            episodeFinished = True
            self.environmentLog.info("Episode %s: No goal reached but %s steps expired!" 
                                     % (self.episodeCounter, self.stepCounter))
        
        # Compute reward
        reward = self.rewardFct(self.state, actionObject['thrust'])
        
        self.trajectoryObservable.addTransition(self._stateForAgent(previousState),
                                                actionObject, reward,
                                                self._stateForAgent(self.state), 
                                                episodeTerminated=episodeFinished)
        
        if episodeFinished:
            self.episodeLengthObservable.addValue(self.episodeCounter,
                                                  self.stepCounter + 1)
            self.returnObservable.addValue(self.episodeCounter,
                                           -self.stepCounter)
             
            self.stepCounter = 0
            self.episodeCounter += 1
            # Reset the simulation to some initial state
            self.state = self.getInitialState()
        
        resultsDict = {"reward" : reward, 
                       "terminalState" : terminalState,                       
                       "nextState" : self._stateForAgent(self.state),
                       "startNewEpisode" : episodeFinished}
        return resultsDict
        
    def stateTransitionFct(self, state, action):
        """ Returns iterator of the successor states of *action* in *state*."""      
        #Applies the action and calculates the new position and velocity
        def minmax (item, limit1, limit2):
            "Bounds item to between limit1 and limit2 (or -limit1)"
            return max(limit1, min(limit2, item))
        
        # Get position and velocity
        position = state["position"]
        velocity = state["velocity"]
        
        # Determine acceleration factor
        if action == 'left': # action is backward thrust
            factor = -1
        elif action == 'none': # action is coast
            factor = 0
        else: # action is forward thrust
            factor = 1
        
        # Do the actual state update
        velocityChange = self.configDict["accelerationFactor"] * factor \
                                - 0.0025 * cos(3 * position)
        velocity = minmax(velocity + velocityChange, -self.maxVelocity,
                          self.maxVelocity)
        position += velocity
        position = minmax(position, self.minPosition, self.maxPosition)
        
        if (position <= self.minPosition) and (velocity < 0):
            velocity = 0.0
            
        if position >= self.goalPosition \
                    and abs(velocity) > self.configDict["maxGoalVelocity"]:
            velocity = -velocity
            
        yield State([position, velocity], [self.stateSpace["position"], 
                                           self.stateSpace["velocity"]]), 1.0
    
    def rewardFct(self, state, action):
        """ Returns the reward obtained after executing *action* in *state*. """
        # We always give reward -1
        return -1
    
    def isTerminalState(self, state):
        """ Returns whether *state* is a terminal state. """
        # Returns whether the car has reached the goal
        return state["position"] >= self.goalPosition \
                and abs(state["velocity"]) <= self.configDict["maxGoalVelocity"] 
    
    def _stateForAgent(self, state):
        return {"position": state["position"]
                            + random.normalvariate(0.0, self.configDict["positionNoise"]),
                "velocity": state["velocity"] 
                            + random.normalvariate(0.0, self.configDict["velocityNoise"])}    
# We assign methods to the environment object that allow the monitor
# to perform some analysis
from mmlf.worlds.mountain_car.environments.optimal_policy_computation \
     import computeOptimalPolicy, getDiscreteStateSpace, scaleState
def getStateSet(self):
    discreteStateSpace = getDiscreteStateSpace(self.configDict["N"])
    return [scaleState(state, self.stateSpace) 
                for state in discreteStateSpace.getStateList()]
def computeOptimalPolicyObj(self):
    return computeOptimalPolicy(self.config)
MountainCarEnvironment.computeOptimalPolicy = computeOptimalPolicyObj
MountainCarEnvironment.getStateSet = getStateSet

EnvironmentClass = MountainCarEnvironment
EnvironmentName = "Mountain Car"

