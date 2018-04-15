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

"""Module that implements the double pole balancing environment and its dynamics."""
# Refactored: 2010-01-16 by Jan Hendrik Metzen

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

from numpy import sin, cos, array, zeros, linspace, fabs, pi
from scipy.integrate import odeint

from mmlf.framework.spaces import StateSpace, ActionSpace
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.environments.single_agent_environment import SingleAgentEnvironment

sgn = lambda x: 0 if x== 0 else x / fabs(x)

class DoublePoleBalancingEnvironment(SingleAgentEnvironment):
    """ The double pole balancing environment
    
    In the double pole balancing environment, the task of the agent is to control
    a cart such that two poles which are mounted on the cart stay in a nearly
    vertical position (to balance them). At the same time, the cart has to stay
    in a confined region.
    
    The agent can apply in every time step a force between -10N and 10N in order to
    accelerate the car. Thus the action space is one-dimensional and continuous. 
    The state consists of the cart's current position and velocity as well as the
    poles' angles and angular velocities. Thus, the state space is six-dimensional
    and continuous.
    
    The config dict of the environment expects the following parameters:
    
    **CONFIG DICT** 
        :GRAVITY: : The gravity force. Benchmark default "-9.8".    
        :MASSCART: : The mass of the cart. Benchmark default "1.0".
        :TAU: : The time step between two commands of the agent. 
                Benchmark default "0.02"                         
        :MASSPOLE_1: : The mass of pole 1. Benchmark default "0.1"
        :MASSPOLE_2: : The mass of pole 2. Benchmark default "0.01"
        :LENGTH_1: : The length of pole 1. Benchmark default "0.5"
        :LENGTH_2: : The length of pole 2. Benchmark default "0.05"
        :MUP: : Coefficient of friction of the poles' hinges.
                Benchmark default "0.000002"
        :MUC: : Coefficient that controls friction. Benchmark default "0.0005"
        :INITIALPOLEANGULARPOSITION1: : Initial angle of pole 1. 
                                        Benchmark default "4.0"
        :MAXCARTPOSITION: : The maximal distance the cart is allowed to move away
                            from its start position. Benchmark default "2.4"
        :MAXPOLEANGULARPOSITION1: : Maximal angle pole 1 is allowed to take on. 
                                    Benchmark default "36.0"
        :MAXPOLEANGULARPOSITION2: : Maximal angle pole 2 is allowed to take on. 
                                    Benchmark default "36.0"
        :MAXSTEPS: : The number of steps the agent must balance the poles. 
                     Benchmark default "100000"
    """
    
    DEFAULT_CONFIG_DICT = {'GRAVITY' : -9.8,    
                           'MASSCART' : 1.0,
                           'TAU' : 0.02,
                           'MASSPOLE_1' : 0.1,
                           'MASSPOLE_2' : 0.1,
                           'LENGTH_1' : 0.5,
                           'LENGTH_2' : 0.05,
                           'MUP' : 0.000002,
                           'MUC' : 0.0005,
                           'INITIALPOLEANGULARPOSITION1' : 4.0,
                           'MAXCARTPOSITION' : 2.4,
                           'MAXPOLEANGULARPOSITION1' : 36.0,
                           'MAXPOLEANGULARPOSITION2' : 36.0,
                           'MAXSTEPS' : 100000}
    
    def __init__(self, useGUI, *args, **kwargs):

        self.environmentInfo = \
            EnvironmentInfo(versionNumber="0.3",
                            environmentName="Double Pole Balancing",
                            discreteActionSpace=False, episodic=True,
                            continuousStateSpace=True,
                            continuousActionSpace=True, stochastic=False)

        super(DoublePoleBalancingEnvironment, self).__init__(useGUI=useGUI, *args, **kwargs)
        
        # Convert from degrees to radians
        self.configDict["INITIALPOLEANGULARPOSITION1"] *= pi/180.0
        self.configDict['MAXPOLEANGULARPOSITION1'] *= pi/180.0
        self.configDict['MAXPOLEANGULARPOSITION2'] *= pi/180.0
        
        # The object which computes the dpb dynamics
        self.dpbDynamics = DoublePoleBalancingDynamics(self.configDict)
        
        #The state space of the Double Pole Balancing Simulation
        oldStyleStateSpace =   {"cartPosition": ("continuous", [(-1.0, 1.0)]),
                                "cartVelocity": ("continuous", [(-0.1, 0.1)]),
                                "poleAngularPosition1": ("continuous", [(-1.0, 1.0)]),
                                "poleAngularVelocity1": ("continuous", [(-0.5, 0.5)]),
                                "poleAngularPosition2": ("continuous", [(-1.0, 1.0)]),
                                "poleAngularVelocity2": ("continuous", [(-0.5, 0.5)])}
        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")
        
        #The action space of the Double Pole Balancing Simulation
        oldStyleActionSpace =  {"force": ("continuous", [(-10, 10)])}
        self.actionSpace = ActionSpace()
        self.actionSpace.addOldStyleSpace(oldStyleActionSpace, limitType="soft")
        
        # The name of the state dimensions that are send to the agent.
        # NOTE: The ordering of the state dimensions is important!
        self.stateNameList = ["cartPosition", "cartVelocity",
                              "poleAngularPosition1", "poleAngularVelocity1",
                              "poleAngularPosition2", "poleAngularVelocity2"]

        # The vector used for normalization of the state for the agent
        self.normalizationVector = array([1.0/self.configDict['MAXCARTPOSITION'],
                                          0.1,
                                          1.0/self.configDict['MAXPOLEANGULARPOSITION1'],
                                          0.2,
                                          1.0/self.configDict['MAXPOLEANGULARPOSITION2'],
                                          0.1])
        
        #The current state of the simulation
        self.initialState = array([0.0, 0.0, 
                                   self.configDict["INITIALPOLEANGULARPOSITION1"],
                                   0.0, 0.0, 0.0])
        #The current state is initially set to the initial state
        self.currentState = array(self.initialState)
        
        if useGUI:
            from mmlf.gui.viewers import VIEWERS
            from mmlf.gui.viewers.trajectory_viewer import TrajectoryViewer
    
            # Add general trajectory viewer
            VIEWERS.addViewer(lambda : TrajectoryViewer(self.stateSpace), 
                              'TrajectoryViewer')
    
    ########################## Interface Functions ##########################
    def getInitialState(self):
        """Returns the initial state of the environment
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
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
        
        force = actionObject['force']
        minForce, maxForce = self.actionSpace['force']['dimensionValues'][0]

        # Force has to be within the allowed range (minForce, maxForce)
        force = min(max(force, minForce), maxForce)

        # if force is less than +/-1/256*10N we set it to this level
        if fabs(force) < 10.0/256:
            force = 10.0/256 if force >= 0 else -10.0/256  

        # Compute the successor state
        self.currentState = self.dpbDynamics.stateTransition(self.currentState,
                                                             force)
        
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
            # Reset the simulation to the initial state (always the same)
            self.currentState = array(self.initialState)
            
            self.trajectoryObservable.addTransition(self._createStateForAgent(previousState),
                                                    actionObject, 1,
                                                    terminalState, 
                                                    episodeTerminated=True)
        else:
            self.stepCounter += 1
            if self.stepCounter in [10000, 20000, 30000, 40000, 50000, 60000, 
                                    70000, 80000, 90000]:
                self.environmentLog.info("Episode %s. Balanced for %s steps!" 
                                         % (self.episodeCounter, self.stepCounter))
                
            self.trajectoryObservable.addTransition(self._createStateForAgent(previousState),
                                                    actionObject, 1,
                                                    self._createStateForAgent(self.currentState), 
                                                    episodeTerminated=False)
        
        resultsDict = {"reward" : 1, #we give always reward 1
                       "terminalState" : terminalState,                       
                       "nextState" : self._createStateForAgent(self.currentState),
                       "startNewEpisode" : episodeFinished}
        return resultsDict
        
    ########################## Helper Functions #####################################
                
    def _checkTerminalState(self):
        """ Returns whether the simulation has reached a terminal state.
        
        A terminal state is reached if the cart or the pole exceed certain 
        boundaries
        """
        return ((fabs(self.currentState[0]) > self.configDict['MAXCARTPOSITION']) 
                or (fabs(self.currentState[2]) > self.configDict['MAXPOLEANGULARPOSITION1'])  
                or (fabs(self.currentState[4]) > self.configDict['MAXPOLEANGULARPOSITION2']))
        
    def _checkEpisodeFinished(self):
        """ Returns whether an episode is finished. 
        
        An episode is finished if a terminal state is reached or the 
        maximum number of steps is exceeded.
        """
        return self._checkTerminalState() \
                or self.stepCounter >= self.configDict["MAXSTEPS"]
    
    def _createStateForAgent(self, state):
        """ Returns the representation of the given *state* for the agent."""
        stateForAgent = dict(zip(self.stateNameList, 
                                 state * self.normalizationVector))
        return stateForAgent


class DoublePoleBalancingDynamics(object):
    """ Class that encapsulates the double pole dynamics computation """
    
    def __init__(self, configDict):
        self.__dict__.update(configDict)
        
        self.ml_1 = self.LENGTH_1 * self.MASSPOLE_1
        self.ml_2 = self.LENGTH_2 * self.MASSPOLE_2

    def _dynamics(self, X, force):
        """ Returns the ODE of the system for force *force* around state *X*."""
        costheta_1 = cos(X[2])
        sintheta_1 = sin(X[2])
        gsintheta_1 = self.GRAVITY * sintheta_1
        costheta_2 = cos(X[4])
        sintheta_2 = sin(X[4])
        gsintheta_2 = self.GRAVITY * sintheta_2
        
        temp_1 = self.MUP * X[3] / self.ml_1
        temp_2 = self.MUP * X[5] / self.ml_2

        fi_1 = (self.ml_1 * X[3]**2 * sintheta_1) + \
               (0.75 * self.MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1))
        fi_2 = (self.ml_2 * X[5]**2 * sintheta_2) + \
               (0.75 * self.MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2))

        mi_1 = self.MASSPOLE_1 * (1.0 - (0.75 * costheta_1 * costheta_1))
        mi_2 = self.MASSPOLE_2 * (1.0 - (0.75 * costheta_2 * costheta_2))

        cartVelocityDot = (force - self.MUC * sgn(X[1]) + fi_1 + fi_2) / \
                                        (mi_1 + mi_2 + self.MASSCART)                                      
        poleAngularVelocity1Dot = -0.75 * (cartVelocityDot * costheta_1 + gsintheta_1 + temp_1) / \
                                       self.LENGTH_1
        poleAngularVelocity2Dot = -0.75 * (cartVelocityDot * costheta_2 + gsintheta_2 + temp_2) / \
                                       self.LENGTH_2
        
        dx_dt = zeros(X.shape)
        dx_dt[0] = X[1] # derivation of cart position is cart velocity 
        dx_dt[1] = cartVelocityDot
        dx_dt[2] = X[3] # derivation of pole1 position is pole1 velocity
        dx_dt[3] = poleAngularVelocity1Dot
        dx_dt[4] = X[5] # derivation of pole2 position is pole2 velocity
        dx_dt[5] = poleAngularVelocity2Dot
        
        return dx_dt
        
    def stateTransition(self, state, force):
        """Computes the successor state of *state* when *force* is applied """
        # Compute differential equation for given force
        dX_dt = lambda X, t: self._dynamics(X, force)
        
        # We are interested in the state at t+self.Tau.
        # We compute 10 intermediate steps
        t = linspace(0, self.TAU, 10)
        succState = odeint(dX_dt, state, t)[-1]
        
        return succState       

    def plot(self, X, t):
        import pylab as p
        cartPos, cartVel, pole1Pos, pole1Vel, pole2Pos, pole2Vel = X.T
        p.plot(t, cartPos, 'r-', label='cartPos')
        p.plot(t, cartVel, 'b-', label='cartVel')
        p.plot(t, pole1Pos, 'y-', label='pole1Pos')
        p.plot(t, pole1Vel, 'k-', label='pole1Vel')
        p.plot(t, pole2Pos, 'g-', label='pole2Pos')
        p.plot(t, pole2Vel, 'c-', label='pole2Vel')
        p.grid()
        p.legend(loc='best')
        p.xlabel('time')
        p.show()

EnvironmentClass = DoublePoleBalancingEnvironment
EnvironmentName = "Double Pole Balancing"
