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

# Author: Jan Hendrik Metzen  (jhm@informatik.uni-bremen.de)
# Created: 2010/10/13
""" Module that contains the the pinball maze environment. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import numpy
import random
import os
from copy import deepcopy

from mmlf.framework.spaces import StateSpace, ActionSpace
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.environments.single_agent_environment import SingleAgentEnvironment

class PinballMazeEnvironment(SingleAgentEnvironment):
    """ The pinball maze environment class.

    The pinball maze environment class.

    .. seealso::
        George Konidaris and Andrew G Barto
        "Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining"
        in "Advances in Neural Information Processing Systems", 2009

    .. versionadded:: 0.9.9

    **CONFIG DICT** 
        :DRAG: : Factor that slows the ball each time step (multiplied to velocity after each step)
        :NOISE: : gaussian noise with MU_POS for position [x,y] and MU_VEL for velocity [xdot,ydot]; as simplification the covariance matrix is just a unit matrix multiplied with SIGMA
        :THRUST_PENALTY: : Reward the agent gains each time it accelerates the ball
        :STEP_PENALTY: : Reward the agent gains each time step it not thrusts or terminates
        :END_EPISODE_REWARD: : Reward the agent gains if the ball reaches the goal
        :SUBSTEPS: : number of dynamic steps of the environment between each of the agent's actions
        :MAZE: : Name of the config file, where the maze is defined. These files are located in folder 'worlds/pinball_maze'
    """
    DEFAULT_CONFIG_DICT = {"DRAG" : 0.995,
                           "NOISE" : {
                               "MU_POS" : [0.0, 0.0],
                               "MU_VEL" : [0.0, 0.0],
                               "SIGMA" : 0.0},
                           "THRUST_PENALTY" : -5,
                           "STEP_PENALTY": -1,
                           "END_EPISODE_REWARD": 10000,
                           "SUBSTEPS": 20,
                           "MAZE": "pinball_simple_single.cfg"}
    
    def __init__(self, useGUI, *args, **kwargs):

        self.environmentInfo = EnvironmentInfo(versionNumber="0.3",
                                               environmentName="PinballMaze",
                                               discreteActionSpace=True,
                                               episodic=True,
                                               continuousStateSpace=True,
                                               continuousActionSpace=False,
                                               stochastic=False)

        super(PinballMazeEnvironment, self).__init__(useGUI=useGUI, *args, **kwargs)
        
        mazeString = open(os.path.dirname(os.path.abspath(__file__))
                             + os.sep + os.pardir + os.sep 
                             + self.configDict['MAZE'], 'r').read()

        #The maze object is created from the description
        self.maze = PinballMaze.createMazeFromString(mazeString)
        
        #The state space of the Maze2d Simulation
        oldStyleStateSpace =   {"x": ("continuous", [(0.0, 1.0)]),
                                "y": ("continuous", [(0.0, 1.0)]),
                                "xdot": ("continuous", [(-1.0, 1.0)]),
                                "ydot": ("continuous", [(-1.0, 1.0)]),
                                }
        
        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")
        
        #The action space of the Maze2d Simulation
        oldStyleActionSpace = {"action": ("discrete", ["xinc", "xdec", "yinc", 
                                                       "ydec", "none"])}
        
        self.actionSpace = ActionSpace()
        self.actionSpace.addOldStyleSpace(oldStyleActionSpace, limitType="soft")
        
        #The current state is initially set to the initial state
        self.currentState = self.getInitialState()
            
        if useGUI:
            # Add viewer specific for the pinball world
            from mmlf.gui.viewers import VIEWERS
            from mmlf.worlds.pinball_maze.environments.pinball_maze_trajectory_viewer \
                        import PinballMazeTrajectoryViewer
            from mmlf.worlds.pinball_maze.environments.pinball_maze_function_viewer \
                        import PinballMazeFunctionViewer
            
            VIEWERS.addViewer(lambda : PinballMazeTrajectoryViewer(self,
                                                                   self.stateSpace),
                              'PinballMaze TrajectoryViewer')
            VIEWERS.addViewer(lambda : PinballMazeFunctionViewer(self,
                                                                 self.stateSpace),
                              'PinballMaze FunctionViewer')
    
    ########################## Interface Functions #####################################
    
    def getInitialState(self):
        """ Returns the initial state of the environment
        
        More information about (valid) states can be found in 
        :ref:`state_and_action_spaces`
        """
        startPos = self.maze.getStartPos()
        return {"x" : startPos[0], "y" : startPos[1], "xdot" : 0.0, "ydot" : 0.0}
    
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
        # The state before executing the action
        previousState = deepcopy(self.currentState)
        
        # Fetch action and do state transition and compute reward
        action = actionObject['action']
        reward, episodeFinished = self._stateTransition(action)
        
        self.accumulatedReward += reward
        
        self.trajectoryObservable.addTransition(previousState, action, 
                                                reward, self.currentState, 
                                                episodeTerminated=episodeFinished)
                      
        terminalState = self.currentState if episodeFinished else None
    
        if episodeFinished:            
            self.episodeLengthObservable.addValue(self.episodeCounter,
                                                  self.stepCounter + 1)
            self.returnObservable.addValue(self.episodeCounter,
                                           self.accumulatedReward)
            self.environmentLog.info("Episode %s lasted for %s steps. Accumulated reward: %s" 
                                        % (self.episodeCounter,self.stepCounter+1,self.accumulatedReward))
            
            self.stepCounter = 0
            self.accumulatedReward = 0.0
            self.episodeCounter += 1
                        
            # Reset to the initial state
            self.currentState = self.getInitialState()
        else:
            self.stepCounter += 1
            if self.stepCounter % 250 == 0: # Keep user up-to-date
                self.environmentLog.info("Episode %s: Agent active for %s steps "
                                         "without reaching goal yet." 
                                                % (self.episodeCounter,
                                                   self.stepCounter))
        
        resultsDict = {"reward" : reward,
                       "terminalState" : terminalState,
                       "nextState" : self._createStateForAgent(self.currentState),
                       "startNewEpisode" : episodeFinished}
        return resultsDict
    
    def _stateTransition(self, action):
        # Determine action effect
        if action == "xinc":
            self.currentState["xdot"] = min(self.currentState["xdot"] + 0.2, 1.0)
        elif action == "xdec":
            self.currentState["xdot"] = max(self.currentState["xdot"] - 0.2, -1.0)
        elif action == "yinc":
            self.currentState["ydot"] = min(self.currentState["ydot"] + 0.2, 1.0)
        elif action == "ydec":
            self.currentState["ydot"] = max(self.currentState["ydot"] - 0.2, -1.0)
        
        # Do state transition, split into SUBSTEPS substeps in order to deal 
        # with collisions on a more fine-granular base.
        for j in range(self.configDict["SUBSTEPS"]):
            # Compute next would-be position
            factor = self.maze.ballRadius / self.configDict["SUBSTEPS"]
            posX = self.currentState["x"] + self.currentState["xdot"] * factor
            posY = self.currentState["y"] + self.currentState["ydot"] * factor
            
            # Check for collision with obstacle
            collided, obstacle = self.maze.collide([posX, posY])
            if collided:
                # Determine collision effect
                self.currentState["xdot"], self.currentState["ydot"] = \
                    self.maze.collisionEffect(oldPos=(self.currentState["x"], 
                                                      self.currentState["y"]),
                                              newPos=(posX, posY),
                                              vel=(self.currentState["xdot"], 
                                                   self.currentState["ydot"]),
                                              obstacle=obstacle)
            else:
                # No collision, go to would-be position
                self.currentState["x"] = posX
                self.currentState["y"] = posY
            
            # Check if target reached
            if self.maze.goalReached(pos=(self.currentState["x"], 
                                          self.currentState["y"])):
                return self.configDict["END_EPISODE_REWARD"], True
            
        # Apply drag
        self.currentState["xdot"] *= self.configDict["DRAG"]
        self.currentState["ydot"] *= self.configDict["DRAG"]
        
        if action == "none":
            return self.configDict["STEP_PENALTY"], False
        else:
            return self.configDict["THRUST_PENALTY"], False
                    
    
    def _createStateForAgent(self, state):
        "Create a state description for the agent"
        mu_pos = self.configDict["NOISE"]["MU_POS"]
        mu_vel = self.configDict["NOISE"]["MU_VEL"]
        sigma = self.configDict["NOISE"]["SIGMA"]
        noisyState = state
        noisyState['x']    = noisyState['x']    + random.gauss(mu_pos[0],sigma)
        noisyState['y']    = noisyState['y']    + random.gauss(mu_pos[1],sigma)
        noisyState['xdot'] = noisyState['xdot'] + random.gauss(mu_vel[0],sigma)
        noisyState['ydot'] = noisyState['ydot'] + random.gauss(mu_vel[1],sigma)
        return noisyState
    
    def plotStateSpaceStructure(self, axis):
        """ Plot structure of state space into given axis. 
        
        Just a helper function for viewers and graphic logging.
        """
        self.maze.draw(axis)
    
    
class PinballMaze(object):
    """ The pinball maze simulator. """
    def __init__(self, startPositions, ballRadius, targetPos, targetRadius, 
                 obstacles):
        self.startPositions = startPositions
        self.ballRadius = ballRadius
        self.targetPos = targetPos
        self.targetRadius = targetRadius
        self.obstacles = obstacles
    
    @staticmethod
    def createMazeFromString(mazeString):
        """ Factory method which creates a maze based on the string which is passed. """
        ballRadius = None,
        targetPos = None,
        targetRadius = None
        startPositions = [] 
        obstacles = []
        for line in map(lambda s: s.strip(), mazeString.split("\n")):
            if line.startswith("ball"):
                ballRadius = float(line.split()[1])
            elif line.startswith("target"):
                targetPos = numpy.array(map(float, line.split()[1:3]))
                targetRadius = float(line.split()[3])
            elif line.startswith("start"):
                tokens = line.split()
                for i in range((len(tokens) - 1)/2):
                    startPositions.append(numpy.array(map(float, tokens[1+i*2:3+i*2])))
            elif line.startswith("polygon"):
                obstacle = []
                tokens = line.split()
                for i in range((len(tokens) - 1)/2):
                    obstacle.append(numpy.array(map(float, tokens[1+i*2:3+i*2])))
                obstacle.append(obstacle[0]) # close poly
                obstacles.append(numpy.array(obstacle))            
                
        return PinballMaze(startPositions, ballRadius, targetPos, targetRadius, 
                           obstacles)
    
    def getStartPos(self):
        return random.choice(self.startPositions)
    
    def collide(self, ballPos):    
        from matplotlib.patches import Circle, PathPatch
        from matplotlib.path import Path    
        for obstacle in self.obstacles:
            obstaclePath = Path(obstacle)
            obstaclePatch = PathPatch(obstaclePath, facecolor=(0.6, 0.6, 0.6),
                                      edgecolor=(0.0, 0.0, 0.0))
            if obstaclePatch.contains_point(ballPos):
                return True, obstaclePatch
        return False, None
    
    
    def collisionEffect(self, oldPos, newPos, vel, obstacle):
        def lineLineIntersect(A,B,C,D):
            # ccw from http://www.bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
            def ccw(A,B,C):
                return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            if ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D):
                # formula from http://local.wasp.uwa.edu.au/~pbourke/geometry/lineline2d/
                ua =    float(((D[0]-C[0])*(A[1]-C[1]))-((D[1]-C[1])*(A[0]-C[0])))/ \
                        float(((D[1]-C[1])*(B[0]-A[0]))-((D[0]-C[0])*(B[1]-A[1])))
                return (A[0]+(ua*(B[0]-A[0])), A[1]+(ua*(B[1]-A[1])))
            return None
        
        def newDirection(pos11, pos12, pos21, pos22):
            vec1 = numpy.array([pos12[0] - pos11[0], pos12[1] - pos11[1]])
            vec2 = numpy.array([pos22[0] - pos21[0], pos22[1] - pos21[1]])
            nvec1 = vec1 / numpy.linalg.norm(vec1)
            nvec2 = vec2 / numpy.linalg.norm(vec2)
            if nvec2[1] < 0 or nvec2[0] == 1:
                nvec2 *= -1
            obstacleAngle = numpy.arccos(nvec2[0])
            movementAngle = numpy.arccos(nvec1[0]) if nvec1[1] > 0 else 2*numpy.pi - numpy.arccos(nvec1[0])
            orientation = 'ccw' if (movementAngle - obstacleAngle) < 0 or \
                                    (movementAngle - obstacleAngle) > numpy.pi else 'cw'
            theta = numpy.arccos(nvec1[0]*nvec2[0] + nvec1[1]*nvec2[1])
            
            if orientation == 'ccw':
                vx = numpy.linalg.norm(vel) * numpy.cos(obstacleAngle + theta)
                vy = numpy.linalg.norm(vel) * numpy.sin(obstacleAngle + theta)
            else:
                vx = numpy.linalg.norm(vel) * numpy.cos(obstacleAngle - theta)
                vy = numpy.linalg.norm(vel) * numpy.sin(obstacleAngle - theta)
            return vx, vy
        
        # Find collision line
        vertices = obstacle.get_path().vertices
        for lineIndex in range(vertices.shape[0] - 1):

            if lineLineIntersect(oldPos, newPos, vertices[lineIndex],
                                 vertices[lineIndex+1]):
                return newDirection(oldPos, newPos, vertices[lineIndex],
                                    vertices[lineIndex+1])
                
    def goalReached(self, pos):
        distToGoal = numpy.linalg.norm(numpy.array(pos) - numpy.array(self.targetPos)) 
        return distToGoal < self.targetRadius
                
    def draw(self, axis):
        from matplotlib.patches import Circle, PathPatch
        from matplotlib.path import Path
                    
        # Plot target
        circle = Circle(self.targetPos, self.targetRadius, facecolor='r')
        axis.add_patch(circle)
        # Plot start positions
        for startPosition in self.startPositions:
            circle = Circle(startPosition, self.ballRadius, facecolor='b')
            axis.add_patch(circle)
            
        # Draw obstacles
        # NOTE: For some reason it seems to be very important that this comes 
        #       -> after <- start and goal drawing!
        for obstacle in self.obstacles:
            obstaclePath = Path(obstacle)
            obstaclePatch = PathPatch(obstaclePath, facecolor=(0.6, 0.6, 0.6),
                                      edgecolor=(0.0, 0.0, 0.0))
            axis.add_patch(obstaclePatch)
        
EnvironmentClass = PinballMazeEnvironment
EnvironmentName = "Pinball 2D"


if __name__ == "__main__":
    mazeString = """
ball 0.015
target 0.5 0.06 0.04
start 0.055 0.95  0.945 0.95

polygon 0.0 0.0 0.0 0.01 1.0 0.01 1.0 0.0 
polygon 0.0 0.0 0.01 0.0 0.01 1.0 0.0 1.0 
polygon 0.0 1.0 0.0 0.99 1.0 0.99 1.0 1.0 
polygon 1.0 1.0 0.99 1.0 0.99 0.0 1.0 0.0 
polygon 0.034 0.852 0.106 0.708 0.33199999999999996 0.674 0.17599999999999996 0.618 0.028 0.718 
polygon 0.15 0.7559999999999999 0.142 0.93 0.232 0.894 0.238 0.99 0.498 0.722 
polygon 0.8079999999999999 0.91 0.904 0.784 0.7799999999999999 0.572 0.942 0.562 0.952 0.82 0.874 0.934 
polygon 0.768 0.814 0.692 0.548 0.594 0.47 0.606 0.804 0.648 0.626 
polygon 0.22799999999999998 0.5760000000000001 0.39 0.322 0.3400000000000001 0.31400000000000006 0.184 0.456 
polygon 0.09 0.228 0.242 0.076 0.106 0.03 0.022 0.178 
polygon 0.11 0.278 0.24600000000000002 0.262 0.108 0.454 0.16 0.566 0.064 0.626 0.016 0.438 
polygon 0.772 0.1 0.71 0.20599999999999996 0.77 0.322 0.894 0.09600000000000002 0.8039999999999999 0.17600000000000002 
polygon 0.698 0.476 0.984 0.27199999999999996 0.908 0.512 
polygon 0.45 0.39199999999999996 0.614 0.25799999999999995 0.7340000000000001 0.438 
polygon 0.476 0.868 0.552 0.8119999999999999 0.62 0.902 0.626 0.972 0.49 0.958 
polygon 0.61 0.014000000000000002 0.58 0.094 0.774 0.05000000000000001 0.63 0.054000000000000006 
polygon 0.33399999999999996 0.014 0.27799999999999997 0.03799999999999998 0.368 0.254 0.7 0.20000000000000004 0.764 0.108 0.526 0.158 
polygon 0.294 0.584 0.478 0.626 0.482 0.574 0.324 0.434 0.35 0.39 0.572 0.52 0.588 0.722 0.456 0.668 
"""

    maze = PinballMaze.createMazeFromString(mazeString)
    import pylab
    maze.draw(pylab.gca())
    pylab.draw()
                
    pylab.draw()
    pylab.show()
