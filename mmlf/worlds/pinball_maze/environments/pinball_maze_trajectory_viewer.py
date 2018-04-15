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


from collections import defaultdict
from itertools import cycle
import numpy

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mmlf.framework.state import State
from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.observables import OBSERVABLES, TrajectoryObservable

class PinballMazeTrajectoryViewer(Viewer):
    
    def __init__(self, pinballMazeEnv, stateSpace):        
        super(PinballMazeTrajectoryViewer, self).__init__()
        
        self.pinballMazeEnv = pinballMazeEnv
        
        self.dimensions = [stateSpace[dimName] for dimName in sorted(stateSpace.keys())]
        
        # The segments that are obtained while drawing is disabled. These
        # segment are drawn one drawing is reenabled 
        self.rememberedSegments = []
        
        # The eval function that can be used for coloring the trajectory
        self.evalFunction = None
        
        self.colorsCycle = cycle([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                                  (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
                                  (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)])
        self.colors = defaultdict(lambda : self.colorsCycle.next())
        self.valueToColorMapping = dict()
        
        # Get required observables
        self.trajectoryObservable = \
                OBSERVABLES.getAllObservablesOfType(TrajectoryObservable)[0]
        
        # Create matplotlib widgets
        plotWidget = QtGui.QWidget(self)
        plotWidget.setMinimumSize(600, 500)
        plotWidget.setWindowTitle("Pinball Maze")
 
        self.fig = Figure((6.0, 5.0), dpi=100)
        self.axis = self.fig.gca()
        self.pinballMazeEnv.plotStateSpaceStructure(self.axis)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(plotWidget)    
        self.canvas.draw()
        
        self.ballPatch = None
        self.linePatches = []
        
        # Add other elements to GUI           
        self.drawingEnabledCheckbox = \
                QtGui.QCheckBox("Drawing enabled", self)
        self.drawingEnabledCheckbox.setChecked(True)
        
        self.drawStyle = "Current Position"
        self.drawStyleLabel = QtGui.QLabel("Draw style")
        self.drawStyleComboBox = QtGui.QComboBox(self)
        self.drawStyleComboBox.addItems(["Current Position", "Last Episode", 
                                         "Online (All)"])
        self.connect(self.drawStyleComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._drawStyleChanged)
                
        self.colorCriterion = "Action"
        self.colorCriterionLabel = QtGui.QLabel("Coloring of trajectory")
        self.colorCriterionComboBox = QtGui.QComboBox(self)
        self.colorCriterionComboBox.addItems(["Action", "Reward", "Q-Value"])
        self.connect(self.colorCriterionComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._colorCriterionChanged) 
                
        # Legend of plot
        self.legendLabel = QtGui.QLabel("Legend:")
        self.legendWidget = QtGui.QListWidget(self)
        
        # Create layout
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidget)
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addWidget(self.drawingEnabledCheckbox)
        self.drawStyleLayout = QtGui.QHBoxLayout()
        self.drawStyleLayout.addWidget(self.drawStyleLabel)
        self.drawStyleLayout.addWidget(self.drawStyleComboBox)
        self.vlayout.addLayout(self.drawStyleLayout)
        self.coloringLayout = QtGui.QHBoxLayout()
        self.coloringLayout.addWidget(self.colorCriterionLabel)
        self.coloringLayout.addWidget(self.colorCriterionComboBox)
        self.vlayout.addLayout(self.coloringLayout)
        self.vlayout.addWidget(self.legendLabel)
        self.vlayout.addWidget(self.legendWidget)
        self.hlayout.addLayout(self.vlayout)
        self.setLayout(self.hlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.trajectoryObservableCallback = \
             lambda *transition: self._updateSamples(*transition)
        self.trajectoryObservable.addObserver(self.trajectoryObservableCallback)
    
    def close(self):
        self.trajectoryObservable.removeObserver(self.trajectoryObservableCallback)
        
        super(PinballMazeTrajectoryViewer, self).close()
                
    def _updateSamples(self, state, action, reward, succState, episodeTerminated):
        # Determine color
        if self.colorCriterion == "Action":
            value = action 
        elif self.colorCriterion == "Reward": 
            value = reward
        elif self.colorCriterion == "Q-Value":
            if self.evalFunction is None: return
            queryState = State((succState['x'], succState['xdot'], 
                                succState['y'], succState['ydot']), 
                               self.dimensions)
            value = self.evalFunction(queryState)
            
            self.minValue = min(value, self.minValue)
            self.maxValue = max(value, self.maxValue)

        if self.drawingEnabledCheckbox.checkState(): # Immediate drawing           
            # Remove ball patch if it is drawn currently
            if self.ballPatch != None:
                self.ballPatch.remove()
                self.ballPatch = None
                
            if self.drawStyle == "Current Position":
                # Remove old trajectory
                self._removeTrajectory()
                self.rememberedSegments = []
                # Plot ball     
                self.ballPatch = Circle([state["x"], state["y"]], 
                                        self.pinballMazeEnv.maze.ballRadius, facecolor='k') 
                self.axis.add_patch(self.ballPatch)
                self.canvas.draw()
            elif self.drawStyle == "Online (All)":   
                # If drawing was just reactivated
                self._drawRememberedSegments()
                # Draw current transition             
                lines = self.axis.plot([state["x"], succState["x"]], 
                                       [state["y"], succState["y"]], '-',
                                       color=self._determineColor(value))
                self.linePatches.extend(lines)
                self.canvas.draw()
            else: # "Last Episode"
                # Remember state trajectory, it will be drawn at the end 
                # of the episode
                self.rememberedSegments.append((state["x"], succState["x"],
                                                state["y"], succState["y"], 
                                                value))
                if episodeTerminated:
                    # Remove last trajectory, draw this episode's trajectory
                    self._removeTrajectory()
                    self._drawRememberedSegments()
                    self.canvas.draw()
                    # When coloring trajectory based on real valued criteria,
                    # we have to update the legend now 
                    if self.colorCriterion == "Q-Value":
                        self.legendWidget.clear()
                        for value in numpy.logspace(0, numpy.log10(self.maxValue - self.minValue + 1), 10):
                            value = value - 1 + self.minValue
                            
                            color = self._determineColor(value)
                            item = QtGui.QListWidgetItem(str(value), self.legendWidget)
                            qColor = QtGui.QColor(int(color[0]*255),
                                                  int(color[1]*255), 
                                                  int(color[2]*255))
                            item.setTextColor(qColor)
                            self.legendWidget.addItem(item) 
        else:
            if self.drawStyle != "Current Position":
                # Remember state trajectory, it will be drawn once drawing is
                # reenabled
                self.rememberedSegments.append((state["x"], succState["x"],
                                                state["y"], succState["y"], 
                                                value))
                
    def _determineColor(self, value):
        # Choose the color for the value
        if self.colorCriterion in ["Action", "Reward"]:
            # Finite number of values 
            if value not in self.valueToColorMapping:
                color = self.colorsCycle.next()
                self.valueToColorMapping[value] = color
                
                # Add to legend
                item = QtGui.QListWidgetItem(str(value), self.legendWidget)
                qColor = QtGui.QColor(int(color[0]*255), int(color[1]*255), 
                                     int(color[2]*255))
                item.setTextColor(qColor)
                self.legendWidget.addItem(item)
                   
            return self.valueToColorMapping[value]
        else:
            if self.maxValue != self.minValue:
                alpha = numpy.log10(value - self.minValue + 1) \
                                / numpy.log10(self.maxValue - self.minValue + 1) 
            else:
                alpha = 0.5
            return (alpha, 0, 1-alpha)
            
    
    def _removeTrajectory(self):
        if len(self.linePatches) > 0:
            for line in self.linePatches:
                line.remove()
            self.linePatches = []
            
    def _drawRememberedSegments(self):
        if len(self.rememberedSegments) > 0:
            for x1, x2, y1, y2, value  in self.rememberedSegments:
                lines = self.axis.plot([x1, x2], [y1, y2], '-', 
                                       color=self._determineColor(value))
                self.linePatches.extend(lines)
            self.rememberedSegments = []
        
    def _drawStyleChanged(self, drawStyle):
        self.drawStyle = drawStyle
        
        if self.drawStyle != "Last Episode" and self.colorCriterion == "Q-Value":
            # This combination is not possible, change coloring criterion
            self._colorCriterionChanged("Reward")
            
            
    def _colorCriterionChanged(self, colorCriterion):
        # If we changed color criterion 
        if colorCriterion != self.colorCriterion:
            # Remove old trajectory
            self._removeTrajectory()
            self.rememberedSegments = []
            self.legendWidget.clear()
            
        self.colorCriterion = colorCriterion
        self.valueToColorMapping = {}

        if self.colorCriterion == "Q-Value":
            # Register to FunctionOverStateSpaceObservable for global Q-Function
            from mmlf.framework.observables import OBSERVABLES, \
                                             FunctionOverStateSpaceObservable
            for functionObservable in OBSERVABLES.getAllObservablesOfType(
                                            FunctionOverStateSpaceObservable):
                # TODO: Name of function is hard-coded
                if functionObservable.title == "Option TopLevel (optimal value function)":
                    def updateEvalFunction(evalFunction):
                        self.evalFunction = evalFunction
                    functionObservable.addObserver(updateEvalFunction)
                    break
                
            # Displaying Q-Value makes only sense when plotting at the end of 
            # an episode
            self._drawStyleChanged("Last Episode")
            
            # We have to remember minimal and maximal value
            self.minValue = numpy.inf
            self.maxValue = -numpy.inf
            
            
                    
