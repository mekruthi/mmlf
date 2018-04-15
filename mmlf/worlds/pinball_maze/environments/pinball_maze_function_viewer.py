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


import threading

import numpy

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.observables import OBSERVABLES, FunctionOverStateSpaceObservable
from mmlf.gui.plot_utils import generate2dStateSlice, generate2dPlotArray

class PinballMazeFunctionViewer(Viewer):
    
    def __init__(self, pinballMazeEnv, stateSpace):        
        super(PinballMazeFunctionViewer, self).__init__()
        
        self.pinballMazeEnv = pinballMazeEnv
        self.stateSpace = stateSpace
        self.actions = []

        self.updateCounter = 0
        self.updatePlotNow = False
        self.evalFunction = None
        
        self.lock = threading.Lock()
        
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
        
        self.plottedPatches = []
                
        # Add a combobox for selecting the function over state space that is observed
        self.selectedFunctionObservable = None
        self.functionObservableLabel = QtGui.QLabel("Function over State Space")
        self.functionObservableComboBox = QtGui.QComboBox(self)
        functionObservables = OBSERVABLES.getAllObservablesOfType(FunctionOverStateSpaceObservable)
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
            self.functionObservableComboBox.addItems([functionObservable.title 
                                                for functionObservable in functionObservables])
            if len(functionObservables) > 0:
                self.selectedFunctionObservable = functionObservables[0].title 
            
        OBSERVABLES.addObserver(updateFunctionObservableBox)
        
        # Slider that controls the granularity of the plot-grid
        self.gridNodesPerDim = 50       
        self.gridNodesSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.gridNodesSlider.setValue(self.gridNodesPerDim)
        self.gridNodesSlider.setMinimum(0)
        self.gridNodesSlider.setMaximum(100)
        self.gridNodesSlider.setTickInterval(10)
        self.gridNodesSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.connect(self.gridNodesSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeGridNodes)
        self.gridNodesLabel = QtGui.QLabel("Grid Nodes Per Dimension: %s" 
                                           % self.gridNodesPerDim )
        
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
        
        # Chosen xvel and yvel values
        self.xVel = 0.5
        self.xVelSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.xVelSlider.setValue(int(self.xVel*10))
        self.xVelSlider.setMinimum(0)
        self.xVelSlider.setMaximum(10)
        self.xVelSlider.setTickInterval(1)
        self.xVelSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.connect(self.xVelSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeXVel)
        self.xVelLabel = QtGui.QLabel("xvel value: %s" % self.xVel)

        self.yVel = 0.5
        self.yVelSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.yVelSlider.setValue(int(self.yVel*10))
        self.yVelSlider.setMinimum(0)
        self.yVelSlider.setMaximum(10)
        self.yVelSlider.setTickInterval(1)
        self.yVelSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.connect(self.yVelSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeYVel)
        self.yVelLabel = QtGui.QLabel("yvel value: %s" % self.xVel)
        
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
        self.gridNodesLayout = QtGui.QHBoxLayout()
        self.gridNodesLayout.addWidget(self.gridNodesLabel)
        self.gridNodesLayout.addWidget(self.gridNodesSlider)
        self.vlayout.addLayout(self.gridNodesLayout)
        self.updateFrequencyLayout = QtGui.QHBoxLayout()
        self.updateFrequencyLayout.addWidget(self.updateFrequencyLabel)
        self.updateFrequencyLayout.addWidget(self.updateFrequencySlider)
        self.vlayout.addLayout(self.updateFrequencyLayout)
        self.vlayout.addWidget(self.updatePlotButton)
        self.xVelLayout = QtGui.QHBoxLayout()
        self.xVelLayout.addWidget(self.xVelLabel)
        self.xVelLayout.addWidget(self.xVelSlider)
        self.vlayout.addLayout(self.xVelLayout)
        self.yVelLayout = QtGui.QHBoxLayout()
        self.yVelLayout.addWidget(self.yVelLabel)
        self.yVelLayout.addWidget(self.yVelSlider)
        self.vlayout.addLayout(self.yVelLayout)
        self.vlayout.addWidget(self.legendLabel)
        self.vlayout.addWidget(self.legendWidget)
        self.hlayout.addLayout(self.vlayout)
        self.setLayout(self.hlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.functionObservable = None
        if self.selectedFunctionObservable:
            self.functionObservable = \
                    OBSERVABLES.getObservable(self.selectedFunctionObservable,
                                              FunctionOverStateSpaceObservable)
            self.functionObservableCallback = \
                 lambda evalFunction: self._updateFunction(evalFunction)
            self.functionObservable.addObserver(self.functionObservableCallback)
            
    def close(self):
        if self.functionObservable is not None:
            # Disconnect from old function observable
            self.functionObservable.removeObserver(self.functionObservableCallback)
        
        super(PinballMazeFunctionViewer, self).close()
        
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
        
        # Check if the observed function returns discrete or continuous value
        discreteFunction = self.functionObservable.discreteValues
        
        # Generate 2d state slice
        defaultDimValues = {"x" : 0, "xdot" : self.xVel, "y" : 0 ,"ydot" : self.yVel}    
        stateSlice = generate2dStateSlice(["x", "y"], self.stateSpace,
                                          defaultDimValues,
                                          gridNodesPerDim=self.gridNodesPerDim)
        
        # Compute values that should be plotted
        values, colorMapping = \
                generate2dPlotArray(self.evalFunction, stateSlice, 
                                    not discreteFunction,
                                    shape=(self.gridNodesPerDim,self.gridNodesPerDim)) 
        
        # If all value are None, we cannot draw anything useful
        if values.mask.all():
            self.lock.release()
            return            
        
        polyCollection = self.axis.pcolor(numpy.linspace(0.0, 1.0, self.gridNodesPerDim), 
                                          numpy.linspace(0.0, 1.0, self.gridNodesPerDim),
                                          values.T)
        self.plottedPatches.append(polyCollection)

        # Set axis limits
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        # Update legend
        self.legendWidget.clear()
        linearSegmentedColorbar = polyCollection.get_cmap()
        if discreteFunction:
            for functionValue, colorValue in colorMapping.items():
                if isinstance(functionValue, tuple):
                    functionValue = functionValue[0] # deal with '(action,)'
                normValue = polyCollection.norm(colorValue)
                if isinstance(normValue, numpy.ndarray): 
                    normValue = normValue[0] # happens when function is constant
                rgbaColor = linearSegmentedColorbar(normValue)
                item = QtGui.QListWidgetItem(str(functionValue), self.legendWidget)
                color = QtGui.QColor(int(rgbaColor[0]*255),
                                     int(rgbaColor[1]*255), 
                                     int(rgbaColor[2]*255))
                item.setTextColor(color)
                self.legendWidget.addItem(item)   
        else:
            for value in numpy.linspace(polyCollection.norm.vmin,
                                        polyCollection.norm.vmax, 10):
                normValue = polyCollection.norm(value)
                if isinstance(normValue, numpy.ndarray): 
                    normValue = normValue[0] # happens when function is constant
                rgbaColor = linearSegmentedColorbar(normValue)
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
        self.functionObservableCallback = \
             lambda evalFunction: self._updateFunction(evalFunction)
        self.functionObservable.addObserver(self.functionObservableCallback)
        
        self.actions = []
        self.lock.release()
                   
    def _changeGridNodes(self):
        self.gridNodesPerDim = self.gridNodesSlider.value()
        self.gridNodesLabel.setText("Grid Nodes Per Dimension: %s" 
                                    % self.gridNodesPerDim )
        # update plot
        self._plotFunction()
            
    def _changeUpdateFrequency(self):
        self.updateFrequency = self.updateFrequencySlider.value() / 100.0
        self.updateFrequencyLabel.setText("UpdateFrequency: %s" 
                                          % self.updateFrequency)

    def _updatePlot(self):
        self.updatePlotNow = True
        
    def _changeXVel(self):
        self.xVel = self.xVelSlider.value() / 10.0
        self.xVelLabel.setText("xvel value: %s" % self.xVel)
        
    def _changeYVel(self):
        self.yVel = self.yVelSlider.value() / 10.0
        self.yVelLabel.setText("yvel value: %s" % self.yVel)
