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

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf.framework.state import State
from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.observables import OBSERVABLES, TrajectoryObservable, \
                                 StateActionValuesObservable

class Maze2DDetailedViewer(Viewer):
    
    def __init__(self, maze, stateSpace, actions):        
        super(Maze2DDetailedViewer, self).__init__()
        
        self.maze = maze
        self.stateSpace = stateSpace
        self.actions = actions
        
        self.samples = defaultdict(lambda : 0)
        self.valueAccessFunction = None
        
        self.redrawRequested = False
        
        # Get required observables
        self.trajectoryObservable = \
                OBSERVABLES.getAllObservablesOfType(TrajectoryObservable)[0]
        self.stateActionValuesObservables = \
                OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable)
    
        # Combo Box for selecting the observable
        self.observableLabel = QtGui.QLabel("Observable")
        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.addItems(map(lambda x: "%s" % x.title, 
                                   self.stateActionValuesObservables))
        self.connect(self.comboBox, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._observableChanged)
        
        # Automatically update combobox when new float stream observables 
        #  are created during runtime
        def updateComboBox(observable, action):
            self.comboBox.clear()
            self.stateActionValuesObservables = \
                    OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable)
            self.comboBox.addItems(map(lambda x: "%s" % x.title, 
                                       self.stateActionValuesObservables))
        OBSERVABLES.addObserver(updateComboBox)
        
        # Combo Box for selecting the updateFrequency
        self.updateFreqLabel = QtGui.QLabel("Update")
        self.updateComboBox = QtGui.QComboBox(self)
        self.updateComboBox.addItems(["Every Episode", "Every Step"])
    
        # Create matplotlib widgets
        plotWidgetPolicy = QtGui.QWidget(self)
        plotWidgetPolicy.setMinimumSize(300, 400)
        plotWidgetPolicy.setWindowTitle ("Policy")
 
        self.figPolicy = Figure((3.0, 4.0), dpi=100)
        self.figPolicy.subplots_adjust(left=0.01, bottom=0.01, right=0.99, 
                                       top= 0.99, wspace=0.05, hspace=0.11)

        self.canvasPolicy = FigureCanvas(self.figPolicy)
        self.canvasPolicy.setParent(plotWidgetPolicy)
        
        ax = self.figPolicy.gca()
        ax.clear()
        self.maze.drawIntoAxis(ax)
        
        self.plotWidgetValueFunction = dict()
        self.figValueFunction = dict()
        self.canvasValueFunction = dict() 
        for index, action in enumerate(self.actions):
            self.plotWidgetValueFunction[action] = QtGui.QWidget(self)
            self.plotWidgetValueFunction[action].setMinimumSize(300, 400)
            self.plotWidgetValueFunction[action].setWindowTitle (str(action))
        
            self.figValueFunction[action] = Figure((3.0, 4.0), dpi=100)
            self.figValueFunction[action].subplots_adjust(left=0.01, bottom=0.01, 
                                                   right=0.99, top= 0.99,
                                                   wspace=0.05, hspace=0.11)
         
            self.canvasValueFunction[action] = FigureCanvas(self.figValueFunction[action])
            self.canvasValueFunction[action].setParent(self.plotWidgetValueFunction[action])
            
            ax = self.figValueFunction[action].gca()
            ax.clear()
            self.maze.drawIntoAxis(ax)
        
        self.textInstances = dict()
        self.arrowInstances = []
               
        self.canvasPolicy.draw()
        for index, action in enumerate(self.actions):
            self.canvasValueFunction[action].draw()
        
        self.mdiArea = QtGui.QMdiArea(self)
        self.mdiArea.addSubWindow(plotWidgetPolicy)
        for index, action in enumerate(self.actions):
            self.mdiArea.addSubWindow(self.plotWidgetValueFunction[action])
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addWidget(self.mdiArea)
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(self.observableLabel)
        self.hlayout.addWidget(self.comboBox)
        self.hlayout.addWidget(self.updateFreqLabel)
        self.hlayout.addWidget(self.updateComboBox)
        self.vlayout.addLayout(self.hlayout)
        self.setLayout(self.vlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.trajectoryObservableCallback = \
             lambda *transition: self.updateSamples(*transition)
        self.trajectoryObservable.addObserver(self.trajectoryObservableCallback)
        
        self.stateActionValuesObservableCallback = \
             lambda valueAccessFunction, actions: self.updateValues(valueAccessFunction, actions)
        if len(self.stateActionValuesObservables) > 0:
            # Show per default the first observable
            self.stateActionValuesObservable = self.stateActionValuesObservables[0] 
            
            self.stateActionValuesObservable.addObserver(self.stateActionValuesObservableCallback)
        else:
            self.stateActionValuesObservable = None
            
    def close(self):
        if self.stateActionValuesObservable is not None:
            # Remove old observable
            self.stateActionValuesObservable.removeObserver(self.stateActionValuesObservableCallback)
        
        super(Maze2DDetailedViewer, self).close()
           
    def updateValues(self, valueAccessFunction, actions):
        self.valueAccessFunction = valueAccessFunction
        # Check if we have to redraw
        if self.redrawRequested or \
                str(self.updateComboBox.currentText()) == "Every Step": 
            self.redraw()
            self.redrawRequested = False
        
    def updateSamples(self, state, action, reward, succState, episodeTerminated):
        state = self.stateSpace.parseStateDict(state)
        self.samples[(state, action)] = self.samples[(state, action)] + 1
        # Check if we have to redraw
        if str(self.updateComboBox.currentText()) == "Every Episode" \
                 and episodeTerminated: 
            self.redrawRequested =True # Request redrawing once the next observable update happens
        
    def redraw(self):
        # Update policy visualization
        for arrow in self.arrowInstances:
            arrow.remove()
        self.arrowInstances = []
        # Iterate over all states and compute the value of the observed function
        dimensions = [self.stateSpace[dimName]
                             for dimName in ["column", "row"]]
        states = [State((column, row), dimensions)#
                    for column in range(self.maze.getColumns())
                        for row in range(self.maze.getRows())]
        for state in states:
            # Evaluate function for this state
            actionValues = dict((action, self.valueAccessFunction(state, (action,)) 
                                        if self.valueAccessFunction is not None else 0.0) 
                                    for action in ["up", "down", "left", "right"])
            maxValue = max(actionValues.values())
            axis = self.figPolicy.gca()
            for action in actionValues.keys():
                if actionValues[action] == maxValue:
                    self._plotArrow(axis, (state[0], state[1]), action) 
        
        # Update Q-function visualization
        for state in states:
            for action in ["up", "down", "left", "right"]:
                value =  self.valueAccessFunction(state, (action,)) \
                                if self.valueAccessFunction is not None else 0.0
                if int(value) == value:
                    valueString =  "%s\n%s" % (int(value), self.samples[(state, action)])
                else:                 
                    valueString =  "%.1f\n%s" % (value, self.samples[(state, action)])
                if (state, action) not in self.textInstances.keys():
                    if isinstance(action, tuple): # For TD-Agents that use crossproduct of action space
                        axis = self.figValueFunction[action[0]].gca()
                    else:
                        axis = self.figValueFunction[action].gca()
                    textInstance = \
                        axis.text(state[0] - 0.3, state[1], valueString, fontsize=8)
                    self.textInstances[(state, action)] = textInstance
                else:
                    self.textInstances[(state, action)].set_text(valueString)

        self.canvasPolicy.draw()
        for index, action in enumerate(self.actions):
            self.canvasValueFunction[action].draw()
            
    def _plotArrow(self, axis, center, direction):
        if isinstance(direction, tuple): # For TD agent with action crossproduct
            direction = direction[0]
        if direction == 'up':
            (dx, dy) = (0.0, 0.6)
        elif direction == 'down':
            (dx, dy) = (0.0, -0.6)
        elif direction == 'right':
            (dx, dy) = (0.6, 0.0)
        elif direction == 'left':
            (dx, dy) = (-0.6, 0.0)
            
        arr = axis.arrow(center[0] - dx/2, center[1] - dy/2, dx, dy,
                         width = 0.05, fc = 'k')
        self.arrowInstances.append(arr)   
        
    def _observableChanged(self, comboBoxIndex):
        if self.stateActionValuesObservable is not None:
            # Remove old observable
            self.stateActionValuesObservable.removeObserver(self.stateActionValuesObservableCallback)
        # Get new observable and add as listener
        self.stateActionValuesObservable = self.stateActionValuesObservables[comboBoxIndex]
        self.stateActionValuesObservable.addObserver(self.stateActionValuesObservableCallback)
