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
# Created: 2008/03/08
""" Two-dimensional maze world environment. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import os
from copy import deepcopy

import mmlf
from mmlf.framework.spaces import StateSpace, ActionSpace
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.environments.single_agent_environment import SingleAgentEnvironment

class Maze2dEnvironment(SingleAgentEnvironment):
    """ The two-dimensional maze environment for an agent without orientation.
    
    A 2d maze world, in which the agent is situated at each moment in time in a 
    certain field (specified by its (row,column) coordinate) and can move
    either upwards, downwards, left or right. The structure of the maze can be
    configured via a text-based config file. 

    **CONFIG DICT**
        :episodesUntilDoorChange: : Episodes that the door will remain in their initial state. After this number of episodes, the door state is inverted.
        :MAZE: : Name of the config file, where the maze is defined. These files are located in folder 'worlds/maze2d'
    
    """
    
    DEFAULT_CONFIG_DICT = {"episodesUntilDoorChange" : 25,
                           "MAZE": "maze_simple.cfg"}
    
    def __init__(self, useGUI, *args, **kwargs):

        self.environmentInfo = EnvironmentInfo(versionNumber="0.3",
                                               environmentName="Maze2D",
                                               discreteActionSpace=True,
                                               episodic=True,
                                               continuousStateSpace=False,
                                               continuousActionSpace=False,
                                               stochastic=False)

        super(Maze2dEnvironment, self).__init__(useGUI=useGUI, *args, **kwargs)
        
        # Reading string which describes the structure of the maze
        mazeDescriptionString = open(mmlf.getRWPath() + os.sep + "config" + os.sep  
                                        + "maze2d" + os.sep + self.configDict['MAZE']).read()
        # Remove comment lines and superfluous whitespace 
        lines = map(lambda line: line.strip(), mazeDescriptionString.split("\n"))
        lines = filter(lambda line: not line.startswith("#"), lines)
        mazeDescriptionString = "\n".join(lines)
                                    
        #The maze object is created from the description
        self.maze = Maze.createMazeFromString(mazeDescriptionString)
        
        #The state space of the Maze2d Simulation
        oldStyleStateSpace =   {"column": ("discrete", range(self.maze.getColumns())),
                                "row": ("discrete", range(self.maze.getRows()))}
        
        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")
        
        #The action space of the Maze2d Simulation
        oldStyleActionSpace =  {"action": ("discrete", ["left", "right", "up", "down"])}
        
        self.actionSpace = ActionSpace()
        self.actionSpace.addOldStyleSpace(oldStyleActionSpace, limitType="soft")
        
        #The current state of the simulation
        self.initialState =  { "row": self.maze.getStartPosition()[0],
                              "column": self.maze.getStartPosition()[1]}
        #The current state is initially set to the initial state
        self.currentState = deepcopy(self.initialState)
                     
        if useGUI:
            from mmlf.gui.viewers import VIEWERS
            from mmlf.gui.viewers.trajectory_viewer import TrajectoryViewer
            from mmlf.worlds.maze2d.environments.maze2d_viewer import Maze2DDetailedViewer
            from mmlf.worlds.maze2d.environments.maze2d_function_viewer import Maze2DFunctionViewer
            
            # Create customized trajectory viewer
            class Maze2dTrajectoryViewer(TrajectoryViewer):
                def __init__(self, stateSpace, plotStateSpaceStructure):
                    super(Maze2dTrajectoryViewer, self).__init__(stateSpace)
                    plotStateSpaceStructure(self.axisTrajectory)
                
            
            VIEWERS.addViewer(lambda : \
                                Maze2dTrajectoryViewer(self.stateSpace,
                                                       lambda ax : self.plotStateSpaceStructure(ax)), 
                              'Maze2dTrajectoryViewer')
            
            # Add viewers for the maze world
            VIEWERS.addViewer(lambda : Maze2DDetailedViewer(self.maze,
                                                            self.stateSpace,
                                                            ["left", "right", "up", "down"]),
                              'Maze2DDetailedViewer')
            VIEWERS.addViewer(lambda : Maze2DFunctionViewer(self.maze,
                                                            self.stateSpace),
                              'Maze2DFunctionViewer')
    
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
        # The state before executing the action
        previousState = dict(self.currentState)
        
        action = actionObject['action']
        # Execute the action which was chosen by the agent
        self._stateTransition(action)
        
        #Check if the episode is finished (i.e. the goal is reached)
        episodeFinished = self._checkEpisodeFinished()
        
        terminalState = self.currentState if episodeFinished else None
        
        if episodeFinished:
            self.episodeLengthObservable.addValue(self.episodeCounter,
                                                  self.stepCounter + 1)
            self.returnObservable.addValue(self.episodeCounter,
                                           -self.stepCounter)
            self.environmentLog.info("Episode %s lasted for %s steps." 
                                     % (self.episodeCounter, self.stepCounter  + 1))
            
            self.stepCounter = 0
            self.episodeCounter += 1
            # Check if maze should be blocked
            if self.episodeCounter == self.configDict["episodesUntilDoorChange"]:
                self.maze.switchBlocking()
            
            # Reset the simulation to the initial state (always the same)
            self.currentState = deepcopy(self.initialState)
            
            self.trajectoryObservable.addTransition(previousState, action, 
                                                    -1, terminalState, 
                                                    episodeTerminated=episodeFinished)
        else:
            self.stepCounter += 1
        
            self.trajectoryObservable.addTransition(previousState, action, 
                                                    -1, self.currentState, 
                                                    episodeTerminated=episodeFinished)
        
        resultsDict = {"reward" : -1, # we always give a reward of -1
                       "terminalState" : terminalState,
                       "nextState" : self._createStateForAgent(self.currentState),
                       "startNewEpisode" : episodeFinished}
        return resultsDict
    
    def _stateTransition(self, action):
        "Execute the specified action and store the resulting state"
        # If the action was move forward:
        currentPos = (self.currentState['row'], self.currentState['column'])
        nextPos = self.maze.tryToMove(currentPos, action)
    
        # The current state is initially set to the initial state
        self.currentState['row'] = nextPos[0]
        self.currentState['column'] = nextPos[1]
        
    def _checkEpisodeFinished(self):
        "Checks whether the episode is finished, i. e. the goal is reached"        
        currentPos = (self.currentState['row'],self.currentState['column'])
        return self.maze.isGoalReached(currentPos)
    
    def _createStateForAgent(self, state):
        "Create a state description for the agent"
        return state
    
    def plotStateSpaceStructure(self, axis):
        """ Plot structure of state space into given axis. 
        
        Just a helper function for viewers and graphic logging.
        """
        self.maze.drawIntoAxis(axis)
    
class Maze(object):
    """
    A class which represents the two-dimensional maze.
    """
    def __init__(self, rows, columns):
        "Create an empty maze with the specified number of rows and columns"
        self.rows = rows
        self.columns = columns
        self.structure = [[None for i in range(columns)] for j in range(rows)]
        self.doors = [[False for i in range(columns)] for j in range(rows)]
        self.startPos = None
        self.goalPos = None
    
    @staticmethod
    def createMazeFromString(mazeDescriptionString):
        """
        Factory method which creates a maze based on the string which is passed.
        """
        structure = []
        for row in map(lambda s: s.strip(), mazeDescriptionString.split('\n')):
            if row == '': continue
            structure.append(list(row))    
            
        maze = Maze(len(structure), len(structure[0]))
        for rowIndex, row in enumerate(structure):
            for colIndex, col in enumerate(row):
                maze.structure[-rowIndex-1][colIndex] = 1 if col == '*' else 0
                if col == 'O': # Blockable fields that are initially open 
                    maze.structure[-rowIndex-1][colIndex] = 0.0
                    maze.doors[-rowIndex-1][colIndex] = True
                elif col == 'C': # Blockable fields that are initially closed 
                    maze.structure[-rowIndex-1][colIndex] = 1.0
                    maze.doors[-rowIndex-1][colIndex] = True  
                elif col == 'S':
                    maze.startPos = (maze.rows - rowIndex - 1,colIndex)
                elif col == 'G':
                    maze.goalPos = (maze.rows - rowIndex - 1,colIndex)
        return maze

    def switchBlocking(self):
        """ Invert traversability of all doors """
        for rowIndex, row in enumerate(self.structure):
            for colIndex, col in enumerate(row):
                if self.doors[rowIndex][colIndex]:
                    if self.structure[rowIndex][colIndex] == 0.0:
                        self.structure[rowIndex][colIndex] = 1.0
                    else:
                        self.structure[rowIndex][colIndex] = 0.0

    def tryToMove(self, pos, orientation):
        """
        The method executes a forward movement starting from the given pos
        in the given direction. If no wall is blocking the movement, the new position
        is returned, otherwise the old position.
        """
        row = pos[0]
        col = pos[1]

        if orientation == 'left':
            newPos = (row, col - 1)
        elif orientation == 'right':
            newPos = (row, col + 1)
        elif orientation == 'up':
            newPos = (row + 1, col)
        elif orientation == 'down':
            newPos = (row - 1, col)
            
        if self.structure[newPos[0]][newPos[1]] == 0:
            return newPos
        else:
            return pos
        
    def getColumns(self):
        "Returns the number of columns of the maze"
        return len(self.structure[0])
        
    def getRows(self):
        "Returns the number of rows of the maze"
        return len(self.structure)
    
    def getStartPosition(self):
        "Returns the start position of the maze"
        return self.startPos
        
    def isGoalReached(self, pos):
        "Checks if the given position is the goal position"
        return pos == self.goalPos
    
    def drawIntoAxis(self, axis):
        """ Draw this maze into the axis. """       
        for y, row in enumerate(self.structure):
            for x, columnValue in enumerate(row):
                if (y, x) == self.startPos:
                    self.plotSquare(axis, (x, y), color='b', zorder=-1)
                elif (y, x) == self.goalPos:
                    self.plotSquare(axis, (x, y), color='r', zorder=-1)
                elif columnValue == 1:
                    self.plotSquare(axis, (x, y), color='k', zorder=2)
#                elif columnValue == 0.5:
#                    self.plotSquare(axis, (x, -y), color='gray', zorder=-1)
                            
    def plotSquare(self, axis, center, color='k', zorder=1):
        return axis.fill([center[0] - 0.5, center[0] + 0.5, center[0] + 0.5, center[0] - 0.5],
                         [center[1] - 0.5, center[1] - 0.5, center[1] + 0.5, center[1] + 0.5],
                         facecolor=color, edgecolor=color, zorder=zorder)
        
EnvironmentClass = Maze2dEnvironment
EnvironmentName = "Maze 2D"

