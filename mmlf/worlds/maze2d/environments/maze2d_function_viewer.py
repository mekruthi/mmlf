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


from itertools import cycle
import threading
import numpy
import pylab
import matplotlib

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf.framework.state import State
from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.observables import OBSERVABLES, FunctionOverStateSpaceObservable, \
                                    StateActionValuesObservable

class Maze2DFunctionViewer(Viewer):
    
    def __init__(self, maze, stateSpace):        
        super(Maze2DFunctionViewer, self).__init__()
        
        self.maze = maze
        self.stateSpace = stateSpace

        self.updateCounter = 0
        self.updatePlotNow = False
        self.evalFunction = None
        
        self.lock = threading.Lock()
        
        # Create matplotlib widgets
        plotWidget = QtGui.QWidget(self)
        plotWidget.setMinimumSize(600, 500)
        plotWidget.setWindowTitle("Maze2D")
 
        self.fig = Figure((6.0, 5.0), dpi=100)
        self.axis = self.fig.gca()
        self.axis.clear()
        self.maze.drawIntoAxis(self.axis)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(plotWidget)    
        self.canvas.draw()
        
        self.plottedPatches = []
                
        # Add a combobox for selecting the function over state space that is observed
        self.selectedFunctionObservable = None
        self.functionObservableLabel = QtGui.QLabel("Function over State Space")
        self.functionObservableComboBox = QtGui.QComboBox(self)
        functionObservables = OBSERVABLES.getAllObservablesOfType(FunctionOverStateSpaceObservable)
        stateActionValueObservables = \
                OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable)
        functionObservables.extend(stateActionValueObservables)
        self.functionObservableComboBox.addItems([functionObservable.title 
                                                 for functionObservable in functionObservables])
        if len(functionObservables) > 0:
            self.selectedFunctionObservable = functionObservables[0].title
                    
        self.connect(self.functionObservableComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._functionObservableChanged)
               
        # Automatically update funtion observable combobox when new observables 
        # are created during runtime
        def updateFunctionObservableBox(viewer, action):
            self.functionObservableComboBox.clear()
            functionObservables = \
                OBSERVABLES.getAllObservablesOfType(FunctionOverStateSpaceObservable)
            stateActionValueObservables = \
                OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable)
            functionObservables.extend(stateActionValueObservables)
            self.functionObservableComboBox.addItems([functionObservable.title 
                                                for functionObservable in functionObservables])
            if self.selectedFunctionObservable is None \
                    and len(functionObservables) > 0:
                self.selectedFunctionObservable = functionObservables[0].title
            else:
                # Let combobox still show the selected observable
                index = self.functionObservableComboBox.findText(self.selectedFunctionObservable)
                if index != -1:
                    self.functionObservableComboBox.setCurrentIndex(index) 
            
        OBSERVABLES.addObserver(updateFunctionObservableBox)
        
        # Add a combobox for for selecting the suboption that is used when
        # a StateActionValuesObservable is observed
        self.selectedSuboption = None
        self.suboptionLabel = QtGui.QLabel("Suboption")
        self.suboptionComboBox = QtGui.QComboBox(self)        

        # Slider that controls the frequency of update the plot
        self.updateFrequency = 0.0      
        self.updateFrequencySlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.updateFrequencySlider.setValue(int(self.updateFrequency * 100))
        self.updateFrequencySlider.setMinimum(0)
        self.updateFrequencySlider.setMaximum(100)
        self.updateFrequencySlider.setTickInterval(0.1)
        self.updateFrequencySlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.connect(self.updateFrequencySlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeUpdateFrequency)
        self.updateFrequencyLabel = QtGui.QLabel("UpdateFrequency: %s" 
                                                 % self.updateFrequency )
        
        # Button to enforce update of plot
        self.updatePlotButton = QtGui.QPushButton("Update Plot")
        self.connect(self.updatePlotButton, QtCore.SIGNAL('clicked()'), 
                     self._updatePlot)        
        
        # Legend of plot
        self.legendLabel = QtGui.QLabel("Legend:")
        self.legendWidget = QtGui.QListWidget(self)

        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidget)
        self.vlayout = QtGui.QVBoxLayout()
        self.functionObservableLayout = QtGui.QHBoxLayout()
        self.functionObservableLayout.addWidget(self.functionObservableLabel)
        self.functionObservableLayout.addWidget(self.functionObservableComboBox)
        self.vlayout.addLayout(self.functionObservableLayout)
        self.suboptionLayout = QtGui.QHBoxLayout()
        self.suboptionLayout.addWidget(self.suboptionLabel)
        self.suboptionLayout.addWidget(self.suboptionComboBox)
        self.vlayout.addLayout(self.suboptionLayout)
        self.updateFrequencyLayout = QtGui.QHBoxLayout()
        self.updateFrequencyLayout.addWidget(self.updateFrequencyLabel)
        self.updateFrequencyLayout.addWidget(self.updateFrequencySlider)
        self.vlayout.addLayout(self.updateFrequencyLayout)
        self.vlayout.addWidget(self.updatePlotButton)
        self.vlayout.addWidget(self.legendLabel)
        self.vlayout.addWidget(self.legendWidget)
        self.hlayout.addLayout(self.vlayout)
        self.setLayout(self.hlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.functionObservable = None
        if self.selectedFunctionObservable is not None:
            self._functionObservableChanged(self.selectedFunctionObservable)
        
    def _updateFunction(self, evalFunction):
        if (self.updateFrequency > 0.0 
                and self.updateCounter > numpy.float(1.0)/self.updateFrequency) \
             or self.updatePlotNow:
            # Reset counter and update plot
            self.updateCounter = 0
            self.updatePlotNow = False
            self.evalFunction = evalFunction  
            self._plotFunction()
        else:        
            # Do not update the plot 
            self.updateCounter += 1

            
    def _plotFunction(self):
        if self.evalFunction is None:
            return
        
        self.lock.acquire()
        
        # Clean up old plot
        for patch in self.plottedPatches:
            patch.remove()
        self.plottedPatches = []
        
        self.colorMapping = dict()
        self.colors = cycle(["b", "g", "r", "c", "m", "y"])
        
        cmap = pylab.get_cmap("jet")
        
        # Check if the observed function returns discrete or continuous value
        discreteFunction = isinstance(self.functionObservable, 
                                      FunctionOverStateSpaceObservable) \
                                and self.functionObservable.discreteValues
        if not discreteFunction:        
            # The values of the observed function over the 2d state space 
            values = numpy.ma.array(numpy.zeros((self.maze.getColumns(),
                                                 self.maze.getRows())),
                                    mask = numpy.zeros((self.maze.getColumns(),
                                                        self.maze.getRows())))
        
        # Iterate over all states and compute the value of the observed function
        dimensions = [self.stateSpace[dimName]
                             for dimName in ["column", "row"]]
        for column in range(self.maze.getColumns()):
            for row in range(self.maze.getRows()):
                # Create state object
                state = State((column, row), dimensions)
                # Evaluate function for this state
                if isinstance(self.functionObservable, 
                              FunctionOverStateSpaceObservable):
                    functionValue = self.evalFunction(state)
                else : # StateActionValuesObservable
                    # Determine chosen option first
                    selectedOption = None
                    for option in self.actions:
                        selectedOptionName = str(self.suboptionComboBox.currentText())
                        if str(option) == selectedOptionName:
                            selectedOption = option
                            break
                    assert selectedOption is not None
                    functionValue = self.evalFunction(state, option)
                            
                # Map function value onto color value
                if discreteFunction:
                    # Deal with situations where the function is only defined over 
                    # part of the state space
                    if functionValue == None or functionValue in [numpy.nan, numpy.inf, -numpy.inf]:
                        continue
                    # Determine color value for function value
                    if not functionValue in self.colorMapping:
                        # Choose value for function value that occurrs for the 
                        # first time
                        self.colorMapping[functionValue] = self.colors.next()
                    patch = self.maze.plotSquare(self.axis, (column, row), 
                                                 self.colorMapping[functionValue])
                    self.plottedPatches.append(patch[0])
                else:
                    # Remember values since we have to know the min and max value
                    # before we can plot
                    values[column, row] = functionValue
                    if functionValue == None or functionValue in [numpy.nan, numpy.inf, -numpy.inf]:
                        values.mask[column, row] = True
        
        # Do the actual plotting for functions with continuous values
        if not discreteFunction:
            minValue = values.min()
            maxValue = values.max()
            for column in range(self.maze.getColumns()):
                for row in range(self.maze.getRows()):
                    if values.mask[column, row]: continue
                    value = (values[column, row] - minValue) / (maxValue - minValue)
                    patch = self.maze.plotSquare(self.axis, (column, row), 
                                                 cmap(value), zorder=0)
                    self.plottedPatches.append(patch[0])
        
        # Set limits
        self.axis.set_xlim(0, len(self.maze.structure[0]) - 1)
        self.axis.set_ylim(0, len(self.maze.structure) - 1)
               
        # Update legend
        self.legendWidget.clear()
        if discreteFunction:
            for functionValue, colorValue in self.colorMapping.items():
                if isinstance(functionValue, tuple):
                    functionValue = functionValue[0] # deal with '(action,)'
                rgbaColor = matplotlib.colors.ColorConverter().to_rgba(colorValue)
                item = QtGui.QListWidgetItem(str(functionValue), self.legendWidget)
                color = QtGui.QColor(int(rgbaColor[0]*255),
                                     int(rgbaColor[1]*255), 
                                     int(rgbaColor[2]*255))
                item.setTextColor(color)
                self.legendWidget.addItem(item)   
        else:
            for value in numpy.linspace(values.min(), values.max(), 10):
                rgbaColor = cmap((value - values.min()) / (values.max() - values.min()))
                item = QtGui.QListWidgetItem(str(value), self.legendWidget)
                color = QtGui.QColor(int(rgbaColor[0]*255),
                                     int(rgbaColor[1]*255), 
                                     int(rgbaColor[2]*255))
                item.setTextColor(color)
                self.legendWidget.addItem(item) 
        
        self.canvas.draw()
        
        self.lock.release()
        
    def _functionObservableChanged(self, selectedFunctionObservable):
        self.lock.acquire()
        if self.functionObservable is not None:
            # Disconnect from old function observable
            self.functionObservable.removeObserver(self.functionObservableCallback)
        
        # Determine new observed function observable
        self.selectedFunctionObservable = str(selectedFunctionObservable)
        
        # Connect to new function observable
        self.functionObservable = OBSERVABLES.getObservable(self.selectedFunctionObservable,
                                                            FunctionOverStateSpaceObservable)
        if self.functionObservable is None: # Observing a StateActionValuesObservable
            self.functionObservable = OBSERVABLES.getObservable(self.selectedFunctionObservable,
                                                                StateActionValuesObservable)
            self.actions = None
            def functionObservableCallback(evalFunction, actions):
                # If we get new options to select from
                if actions != self.actions: 
                    # Update suboptionComboBox 
                    self.actions = actions
                    self.suboptionComboBox.clear()
                    self.suboptionComboBox.addItems([str(action) for action in actions])
                self._updateFunction(evalFunction)                    
                
            self.functionObservableCallback = functionObservableCallback
            self.functionObservable.addObserver(self.functionObservableCallback)
        else: # Observing a FunctionOverStateSpaceObservable
            self.functionObservableCallback = \
                 lambda evalFunction: self._updateFunction(evalFunction)
            self.functionObservable.addObserver(self.functionObservableCallback)
        
        self.lock.release()
            
    def _changeUpdateFrequency(self):
        self.updateFrequency = self.updateFrequencySlider.value() / 100.0
        self.updateFrequencyLabel.setText("UpdateFrequency: %s" 
                                          % self.updateFrequency)

    def _updatePlot(self):
        self.updatePlotNow = True
