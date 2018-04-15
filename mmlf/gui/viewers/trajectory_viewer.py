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


import itertools
from collections import deque

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.observables import OBSERVABLES, TrajectoryObservable

class TrajectoryViewer(Viewer):
    
    def __init__(self, stateSpace):        
        super(TrajectoryViewer, self).__init__()
        
        self.stateSpace = stateSpace
        
        # Define colors for plotting
        self.colors = itertools.cycle(['g', 'b', 'c', 'm', 'k', 'y'])
                
        self.plotLines = deque()
        
        # Combo Boxes for selecting displaced state space dimensions
        self.comboBox1 = QtGui.QComboBox(self)
        self.comboBox1.addItems(sorted(stateSpace.keys()))
        self.comboBox2 = QtGui.QComboBox(self)
        self.comboBox2.addItems(sorted(stateSpace.keys()))
        self.comboBox2.setCurrentIndex(1)
        self.dimension1 = sorted(self.stateSpace.keys())[0]
        self.dimension2 = sorted(self.stateSpace.keys())[1]
        self.connect(self.comboBox1, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._dimension1Changed)
        self.connect(self.comboBox2, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._dimension2Changed)
        
        # Slider for controlling the number of Trajectories
        self.maxLines = 5
        self.numberTrajectoriesSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.numberTrajectoriesSlider.setValue(self.maxLines)
        self.numberTrajectoriesSlider.setMinimum(1)
        self.numberTrajectoriesSlider.setMaximum(20)
        self.numberTrajectoriesSlider.setTickInterval(5)
        self.numberTrajectoriesSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        
        self.connect(self.numberTrajectoriesSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeNumTrajectories)
        
        # Checkbox for dis-/enabling trajectory plotting
        self.plottingEnabled = True
        self.enabledCheckBox = QtGui.QCheckBox("Plotting Enabled")
        self.enabledCheckBox.setChecked(self.plottingEnabled)
        self.connect(self.enabledCheckBox, QtCore.SIGNAL('stateChanged(int)'), 
                     self._enablingPlotting)
        
        # Some labels
        self.dimension1Label = QtGui.QLabel("Dimension X Axis")
        self.dimension2Label = QtGui.QLabel("Dimension Y Axis")
        self.numTrajectoriesLabel = QtGui.QLabel("Trajectories shown")
        
        # Create matplotlib widgets
        plotWidgetTrajectory = QtGui.QWidget(self)
        plotWidgetTrajectory.setMinimumSize(800, 500)
 
        self.figTrajectory = Figure((8.0, 5.0), dpi=100)
        self.axisTrajectory = self.figTrajectory .gca()
        self.canvasTrajectory = FigureCanvas(self.figTrajectory)
        self.canvasTrajectory .setParent(plotWidgetTrajectory )
        
        # Initialize plotting
        self._reinitializePlot()
        
        # Create layout
        self.vlayout = QtGui.QVBoxLayout()
        self.hlayout1 = QtGui.QHBoxLayout()
        self.hlayout1.addWidget(self.dimension1Label)
        self.hlayout1.addWidget(self.comboBox1)
        self.hlayout2 = QtGui.QHBoxLayout()
        self.hlayout2.addWidget(self.dimension2Label)
        self.hlayout2.addWidget(self.comboBox2)
        self.hlayout3 = QtGui.QHBoxLayout()
        self.hlayout3.addWidget(self.numTrajectoriesLabel)
        self.hlayout3.addWidget(self.numberTrajectoriesSlider)
        
        self.vlayout.addLayout(self.hlayout1)
        self.vlayout.addLayout(self.hlayout2)
        self.vlayout.addLayout(self.hlayout3)
        self.vlayout.addWidget(self.enabledCheckBox)
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidgetTrajectory)
        self.hlayout.addLayout(self.vlayout)
        self.setLayout(self.hlayout)
        
        # Connect to trajectory observable
        self.trajectoryObservable = \
                OBSERVABLES.getAllObservablesOfType(TrajectoryObservable)[0]
        self.trajectoryObservableCallback = \
             lambda *transition: self.addTransition(*transition)
        self.trajectoryObservable.addObserver(self.trajectoryObservableCallback)
        
    def close(self):
        self.trajectoryObservable.removeObserver(self.trajectoryObservableCallback)
        
        super(TrajectoryViewer, self).close()
    
    def addTransition(self, state, action, reward, succState, episodeTerminated):
        if not self.plottingEnabled: return
        if len(self.dim1Values) == 0: # for the start state
            self.dim1Values.append(state[self.dimension1])
            self.dim2Values.append(state[self.dimension2])
        self.dim1Values.append(succState[self.dimension1])
        self.dim2Values.append(succState[self.dimension2])
        if episodeTerminated:
            plotLine = self.axisTrajectory.plot(self.dim1Values, self.dim2Values, 
                                                self.colors.next())[0]
            self.plotLines.append(plotLine)
            while len(self.plotLines) > self.maxLines:
                # Remove the oldest line
                oldestLine = self.plotLines.popleft()
                oldestLine.remove()
            
            if self.stateSpace[self.dimension1].isContinuous():
                self.axisTrajectory.set_xlim(self.stateSpace[self.dimension1]["dimensionValues"][0])
            else:
                self.axisTrajectory.set_xlim(min(self.stateSpace[self.dimension1]["dimensionValues"]),
                                             max(self.stateSpace[self.dimension1]["dimensionValues"]))
            if self.stateSpace[self.dimension2].isContinuous():
                self.axisTrajectory.set_ylim(self.stateSpace[self.dimension2]["dimensionValues"][0])
            else:
                self.axisTrajectory.set_ylim(min(self.stateSpace[self.dimension2]["dimensionValues"]),
                                             max(self.stateSpace[self.dimension2]["dimensionValues"]))
            self.canvasTrajectory.draw()
            self.dim1Values = []
            self.dim2Values = [] 

    def _reinitializePlot(self):
        self.dim1Values = []
        self.dim2Values = []
        for line in self.plotLines:
            line.remove()
        self.plotLines = deque()
        if self.stateSpace[self.dimension1].isContinuous():
            self.axisTrajectory.set_xlim(self.stateSpace[self.dimension1]["dimensionValues"][0])
        else:
            self.axisTrajectory.set_xlim(min(self.stateSpace[self.dimension1]["dimensionValues"]),
                                         max(self.stateSpace[self.dimension1]["dimensionValues"]))
        if self.stateSpace[self.dimension2].isContinuous():
            self.axisTrajectory.set_ylim(self.stateSpace[self.dimension2]["dimensionValues"][0])
        else:
            self.axisTrajectory.set_ylim(min(self.stateSpace[self.dimension2]["dimensionValues"]),
                                         max(self.stateSpace[self.dimension2]["dimensionValues"]))
        self.canvasTrajectory.draw()    
    
    def _dimension1Changed(self, comboBoxIndex):
        self.dimension1 = sorted(self.stateSpace.keys())[comboBoxIndex]
        self._reinitializePlot()
        
    def _dimension2Changed(self, comboBoxIndex):
        self.dimension2 = sorted(self.stateSpace.keys())[comboBoxIndex]
        self._reinitializePlot()
        
    def _changeNumTrajectories(self):
        self.maxLines = self.numberTrajectoriesSlider.value()
        
    def _enablingPlotting(self):
        if self.enabledCheckBox.isChecked():
            self.plottingEnabled = True
            self._reinitializePlot()
        else:
            self.plottingEnabled = False
