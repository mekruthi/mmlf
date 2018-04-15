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
from mmlf.framework.observables import OBSERVABLES, StateActionValuesObservable, \
                                            TrajectoryObservable
from mmlf.gui.plot_utils import generate2dStateSlice, generate2dPlotArray

class MountainCarValueFunctionViewer(Viewer):
    
    def __init__(self, stateSpace):        
        super(MountainCarValueFunctionViewer, self).__init__()
        self.stateSpace = stateSpace
        
        self.lock = threading.Lock()
        
        # Add a combobox for selecting the policy observable
        self.valueFunctionObservableLabel = QtGui.QLabel("Value Function Observable")
        self.valueFunctionObservableComboBox = QtGui.QComboBox(self)
        valueFunctionObservables = OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable)
        self.valueFunctionObservableComboBox.addItems([valueFunctionObservable.title 
                                                 for valueFunctionObservable in valueFunctionObservables])
        if len(valueFunctionObservables) > 0:
            self.selectedValueFunctionObservable = valueFunctionObservables[0].title
        else:
            self.selectedValueFunctionObservable = None
        
        self.connect(self.valueFunctionObservableComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._valueFunctionObservableChanged) 
        
        # Automatically update policy observable combobox when new observables 
        # are created during runtime
        def updateValueFunctionObservableBox(viewer, action):
            self.valueFunctionObservableComboBox.clear()
            valueFunctionObservables = OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable)
            self.valueFunctionObservableComboBox.addItems([valueFunctionObservable.title 
                                                for valueFunctionObservable in valueFunctionObservables])
            self.selectedValueFunctionObservable = valueFunctionObservables[0].title 
            
        OBSERVABLES.addObserver(updateValueFunctionObservableBox)
        
        # Get trajectory observable which is required for informing about end of episode
        self.trajectoryObservable = \
                OBSERVABLES.getAllObservablesOfType(TrajectoryObservable)[0]
        self.episodeTerminated = False

        # Slider that controls the granularity of the plot-grid
        self.gridNodesPerDim = 25        
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
                
        # Create matplotlib widgets
        plotWidget = QtGui.QWidget(self)
        plotWidget.setMinimumSize(1000, 700)
        plotWidget.setWindowTitle("Value Function")
 
        self.fig = Figure((10.0, 7.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(plotWidget)    
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidget)
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addWidget(self.valueFunctionObservableLabel)
        self.vlayout.addWidget(self.valueFunctionObservableComboBox)
        self.vlayout.addWidget(self.gridNodesLabel)
        self.vlayout.addWidget(self.gridNodesSlider)
        self.hlayout.addLayout(self.vlayout)
        
        self.setLayout(self.hlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.trajectoryObservableCallback = \
             lambda *transition: self.updateSamples(*transition)
        self.trajectoryObservable.addObserver(self.trajectoryObservableCallback)
        
        self.valueFunctionObservable = None
        if self.selectedValueFunctionObservable != None:
            self.valueFunctionObservable = OBSERVABLES.getObservable(self.selectedValueFunctionObservable,
                                                                     StateActionValuesObservable)
            self.valueFunctionObservableCallback = \
                 lambda valueAccessFunction, actions: self.updateValueFunction(valueAccessFunction, actions)
            self.valueFunctionObservable.addObserver(self.valueFunctionObservableCallback)
            
    def close(self):
        self.trajectoryObservable.removeObserver(self.trajectoryObservableCallback)
        if self.valueFunctionObservable:
            self.valueFunctionObservable.removeObserver(self.valueFunctionObservableCallback)
        
        super(MountainCarValueFunctionViewer, self).close()
                
    def updateSamples(self, state, action, reward, succState, episodeTerminated):
        if episodeTerminated:
            self.episodeTerminated = True
                
    def updateValueFunction(self, valueAccessFunction, actions):
        # We only update the plot at the end of an episode
        if not self.episodeTerminated: return
        self.episodeTerminated = False
        
        self.lock.acquire()
        
        # Generate 2d state slice
        defaultDimValues = {"position" : 0, "velocity" : 0}    
        stateSlice = generate2dStateSlice(["position", "velocity"], 
                                          self.stateSpace, defaultDimValues,
                                          gridNodesPerDim=self.gridNodesPerDim)

        rows = int(numpy.ceil((len(actions) +1)/2.0))
        # For all actions that should be plotted
        self.fig.clear()
        self.fig.subplots_adjust(left=0.02, right=0.97, bottom=0.05, top=0.95)
        self.contours = []
        maxValue, minValue = -numpy.inf, numpy.inf
        for plotNum, action in enumerate(actions):
            # Clear old plot
            subplot = self.fig.add_subplot(rows, 2, plotNum + 1)
            subplot.clear()        
            
            # Compute values that should be plotted
            values, colorMapping = \
                    generate2dPlotArray(lambda state: valueAccessFunction(state, action),
                                        stateSlice, continuousFunction=True,
                                        shape=(self.gridNodesPerDim, self.gridNodesPerDim)) 
            
            if values.min() == values.max(): continue
            
            # Plot data
            self.contours.append(subplot.contour(numpy.linspace(0.0, 1.0, self.gridNodesPerDim),
                                                 numpy.linspace(0.0, 1.0, self.gridNodesPerDim),
                                                 values, 50, linewidths=0.5,
                                                 colors='k'))
            contour = subplot.contourf(numpy.linspace(0.0, 1.0, self.gridNodesPerDim),
                                       numpy.linspace(0.0, 1.0, self.gridNodesPerDim),
                                       values, 50)
            self.contours.append(contour)
            self.fig.colorbar(contour)
            
            maxValue = max(maxValue, values.max())
            minValue = min(minValue, values.min())

            # Labeling etc.
            subplot.set_xlim(0, 1)
            subplot.set_ylim(0, 1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title(action)
            subplot.legend()

        for contour in self.contours:
            contour.set_clim(vmin=minValue, vmax=maxValue)

        self.canvas.draw()        
        self.lock.release()
        
    def _valueFunctionObservableChanged(self, selectedValueFunctionObservable):
        self.lock.acquire()
        # Disconnect from old observable
        if self.valueFunctionObservable:
            self.valueFunctionObservable.removeObserver(self.valueFunctionObservableCallback)
        
        # Determine new observed policy observable
        self.selectedValueFunctionObservable = str(selectedValueFunctionObservable)
        
        # Connect to new policy observable
        self.valueFunctionObservable = OBSERVABLES.getObservable(self.selectedValueFunctionObservable,
                                                                 StateActionValuesObservable)
        self.valueFunctionObservableCallback = \
             lambda valueAccessFunction, actions: self.updateValueFunction(valueAccessFunction, actions)
        self.valueFunctionObservable.addObserver(self.valueFunctionObservableCallback)
        
        self.lock.release()
          
    
    def _changeGridNodes(self):
        self.gridNodesPerDim = self.gridNodesSlider.value()
        self.gridNodesLabel.setText("Grid Nodes Per Dimension: %s" 
                                    % self.gridNodesPerDim )

