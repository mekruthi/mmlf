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

"""Module that implements the single pole balancing environment and its dynamics."""
# This module contains an implementation of the single pole balancing dynamics which can be 
# used in a world of the mmlf.framework.

import math
from copy import deepcopy

from mmlf.framework.spaces import StateSpace, ActionSpace
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.environments.single_agent_environment import SingleAgentEnvironment

class SinglePoleBalancingEnvironment(SingleAgentEnvironment):
    """ The single pole balancing environment.
    
    In the single pole balancing environment, the task of the agent is to control
    a cart such that a pole which is mounted on the cart stays in a nearly
    vertical position (to balance it). At the same time, the cart has to stay
    in a confined region.
    
    The agent can apply in every time step a force between -2N and 2N in order to
    accelerate the car. Thus the action space is one-dimensional and continuous. 
    The state consists of the cart's current position and velocity as well as the
    pole's angle and angular velocity. Thus, the state space is four-dimensional
    and continuous.
    
    **CONFIG DICT** 
        :GRAVITY: : The gravity force. Benchmark default "-9.8"    
        :MASSCART: : The mass of the cart. Benchmark default "1.0"
        :MASSPOLE: : The mass of the pole. Benchmark default "0.1"
        :TOTAL_MASS: : The total mass (pole + cart). Benchmark default "1.1"
        :LENGTH: : The length of the pole. Benchmark default "0.5"
        :POLEMASS_LENGTH: : The center of mass of the pole. Benchmark default "0.05"
        :TAU: : The time step between two commands of the agent. Benchmark default "0.02"                         
        :MAXCARTPOSITION: : The maximal distance the cart is allowed to move away
                            from its start position. Benchmark default "7.5"
        :MAXPOLEANGULARPOSITION: : Maximal angle the pole is allowed to take on. Benchmark default "0.7"
        :MAXSTEPS: : The number of steps the agent must balance the poles. Benchmark default "100000"
    """
    
    DEFAULT_CONFIG_DICT = {'GRAVITY' : 9.8,    
                           'MASSCART' : 1.0,
                           'MASSPOLE' : 0.1,
                           'TOTAL_MASS' : 1.1,
                           'LENGTH' : 0.5,
                           'POLEMASS_LENGTH' : 0.05,
                           'TAU' : 0.02,                         
                           'MAXCARTPOSITION' : 7.5,
                           'MAXPOLEANGULARPOSITION' : 0.7,
                           'MAXSTEPS' : 100000}
    
    def __init__(self, useGUI, *args, **kwargs):
        
        self.environmentInfo = EnvironmentInfo(versionNumber="0.3",
                                               environmentName="Single Pole Balancing",
                                               discreteActionSpace=False,
                                               episodic=True,
                                               continuousStateSpace=True,
                                               continuousActionSpace=True,
                                               stochastic=False)

        super(SinglePoleBalancingEnvironment, self).__init__(useGUI=useGUI, *args, **kwargs)
        
        #The state space of the Single Pole Balancing Simulation
        oldStyleStateSpace = {"cartPosition": ("continuous", [(-3.125, 3.125)]),
                              "cartVelocity": ("continuous", [(-0.5, 0.5)]),
                              "poleAngularPosition": ("continuous", [(-1.13, 1.13)]),
                              "poleAngularVelocity": ("continuous", [(-0.80, 0.80)]),
                              }
        
        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")
        
        #The action space of the Single Pole Balancing Simulation
        oldStyleActionSpace =  {"force": ("continuous", [(-2, 2)])}
        
        self.actionSpace = ActionSpace()
        self.actionSpace.addOldStyleSpace(oldStyleActionSpace, limitType="soft")

        #The current state of the simulation
        #Note that the values of this dict can be accesed directly as 
        #attributes of the class (see the __getattr__ and _setattr__ method)
        self.initialState =  { 
                     "cartPosition": 0.0,
                     "poleAngularPosition": 0.1,
                     "cartVelocity": 0.0,
                     "poleAngularVelocity": 0.0,
                  }
        #The current state is initially set to the initial state
        self.currentState = deepcopy(self.initialState)
        
        if useGUI:
            from mmlf.gui.viewers import VIEWERS
            from mmlf.gui.viewers.trajectory_viewer import TrajectoryViewer
            from mmlf.worlds.single_pole_balancing.environments.spb_trajectory_viewer import SPBTrajectoryViewer
            
            # Add general trajectory viewer
            VIEWERS.addViewer(lambda : TrajectoryViewer(self.stateSpace), 
                              'TrajectoryViewer')
            
            VIEWERS.addViewer(lambda : SPBTrajectoryViewer(),
                              'SPB Cart Viewer')
             
    def __setattr__(self, attrName, attrValue):
        """
        Sets the attribute with name attrName to the Value attrValue.
        
        If there is no such attribute but a key with this name exists in self.currentState,
        this entry of the dictiionary is updated instead.
        """
        if attrName in self.__dict__.iterkeys(): 
            self.__dict__[attrName] = attrValue
        elif attrName != 'currentState' \
             and hasattr(self,'currentState') \
             and attrName in self.currentState.iterkeys():
            self.currentState[attrName] = attrValue
        else:
            self.__dict__[attrName] = attrValue
                
    def __getattr__(self, attrName):
        """
        Returns the value of the attribute specified by attrName. If there is no such attribute,
        it checks if such an attribute is contained in the self.currentState dict.
        """
        if attrName in self.__dict__.iterkeys(): 
            return self.__dict__[attrName]
        elif attrName != 'currentState' and attrName in self.currentState.iterkeys():
            return self.currentState[attrName]
        else:
            raise AttributeError("%s object has no attribute %s" % (self.__class__.__name__, attrName))
    
    ########################## Interface Functions #####################################
    def getInitialState(self):
        """ Returns the initial state of the environment """
        return self._createStateForAgent(self.initialState)
    
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
        previousState = self.currentState
        
        # Determine force applied to the cart
        force = actionObject['force'] # wish be the agent
        force = self.actionSpace.chopContinuousAction(force) # actual force
        
        self._stateTransition(force)
        
        episodeFinished = self._checkEpisodeFinished()
        terminalState = self._createStateForAgent(self.currentState) \
                                 if episodeFinished else None
        
        if episodeFinished:
            self.episodeLengthObservable.addValue(self.episodeCounter,
                                                  self.stepCounter + 1)
            self.returnObservable.addValue(self.episodeCounter,
                                           self.stepCounter)
            self.environmentLog.info("Episode %s lasted for %s steps." 
                                     % (self.episodeCounter, self.stepCounter  + 1))
            
            self.stepCounter = 0
            self.episodeCounter += 1
            #Reset the environment to the initial state (always the same)
            self.currentState = deepcopy(self.initialState)
            
            self.trajectoryObservable.addTransition(self._createStateForAgent(previousState),
                                                    actionObject, 1,
                                                    terminalState, 
                                                    episodeTerminated=True)
        else:
            self.stepCounter += 1
            if self.stepCounter in [10000, 20000, 30000, 40000, 50000, 60000, 
                                    70000, 80000, 90000]:
                self.environmentLog.info("Balanced for %s steps!" % self.stepCounter)
            
            self.trajectoryObservable.addTransition(self._createStateForAgent(previousState),
                                                    actionObject, 1,
                                                    self._createStateForAgent(self.currentState), 
                                                    episodeTerminated=False)
        
        resultsDict = {"reward" : 1, # we always give a reward of 1
                       "terminalState" : terminalState,
                       "nextState" : self._createStateForAgent(self.currentState),
                       "startNewEpisode" : episodeFinished}
        return resultsDict
        
    ########################## Helper Functions #####################################
    def _stateTransition(self, force):
        "Update self.currentState with new values based on the current values and the force applied"
        costheta = math.cos(self.currentState["poleAngularPosition"])
        sintheta = math.sin(self.currentState["poleAngularPosition"])
        
        temp = (force + self.configDict["POLEMASS_LENGTH"] * self.currentState["poleAngularPosition"] * \
               self.currentState["poleAngularPosition"] * sintheta)/ self.configDict["TOTAL_MASS"]

        thetaacc = (self.configDict["GRAVITY"] * sintheta - costheta* temp)/ \
                   (self.configDict["LENGTH"] * (1.333333333333 \
                   - self.configDict["MASSPOLE"] * costheta * costheta / self.configDict["TOTAL_MASS"]))

        xacc  = temp - self.configDict["POLEMASS_LENGTH"] * thetaacc* costheta / self.configDict["TOTAL_MASS"]
        
        #Update the four state variables, using Euler's method. 
        
        self.currentState["cartPosition"] = self.currentState["cartPosition"] + self.configDict["TAU"] * self.currentState["cartVelocity"]
        self.currentState["cartVelocity"] = self.currentState["cartVelocity"] + self.configDict["TAU"] * xacc
        self.currentState["poleAngularPosition"] = self.currentState["poleAngularPosition"] + self.configDict["TAU"] * self.currentState["poleAngularVelocity"]
        self.currentState["poleAngularVelocity"] = self.currentState["poleAngularVelocity"] + self.configDict["TAU"] * thetaacc
    
    def _checkTerminalState(self):
        """
        Returns whether the simulation has reached a terminal state.
        
        A terminal state is reached if the cart or the pole exceed certain boundaries
        """
        return ((math.fabs(self.currentState["cartPosition"]) > self.configDict["MAXCARTPOSITION"]) \
                    or (math.fabs(self.currentState["poleAngularPosition"]) > self.configDict["MAXPOLEANGULARPOSITION"]))
        
    def _checkEpisodeFinished(self):
        """
        Returns whether an episode is finished. 
        
        An episode is finished if a terminal state is reached or the maximum number of steps is exceeded.
        """
        return self._checkTerminalState() or self.stepCounter >= self.configDict["MAXSTEPS"]-1
    
    def _createStateForAgent(self, state):
        """
        Creates a subset of the state which can be communicated to an agent
        """
        # pass the state along unmodified
        stateForAgent =   {
                         "cartPosition": state['cartPosition']/2.4,
                         "cartVelocity": state['cartVelocity']/10.0,
                         "poleAngularPosition": state['poleAngularPosition']/0.62,
                         "poleAngularVelocity": state['poleAngularVelocity']/5.0,
                        }
        return stateForAgent
          
EnvironmentClass = SinglePoleBalancingEnvironment
EnvironmentName = "Single Pole Balancing"
