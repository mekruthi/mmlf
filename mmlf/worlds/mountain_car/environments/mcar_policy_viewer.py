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
import matplotlib

from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.observables import OBSERVABLES, FunctionOverStateSpaceObservable, \
                                            TrajectoryObservable
from mmlf.gui.plot_utils import generate2dStateSlice, generate2dPlotArray

class MountainCarPolicyViewer(Viewer):
    
    def __init__(self, stateSpace):        
        super(MountainCarPolicyViewer, self).__init__()
        self.stateSpace = stateSpace
        self.actions = []
        self.colors = ['r','g','b', 'c', 'y']
        
        self.lock = threading.Lock()
        
        # Add a combobox for selecting the policy observable
        self.policyObservableLabel = QtGui.QLabel("Policy Observable")
        self.policyObservableComboBox = QtGui.QComboBox(self)
        policyObservables = \
            OBSERVABLES.getAllObservablesOfType(FunctionOverStateSpaceObservable)
        self.policyObservableComboBox.addItems([policyObservable.title 
                                                 for policyObservable in policyObservables])
        self.selectedPolicyObservable = None
        if len(policyObservables) > 0:
            self.selectedPolicyObservable = policyObservables[0].title
        
        self.connect(self.policyObservableComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._policyObservableChanged) 
        
        # Automatically update policy observable combobox when new observables 
        # are created during runtime
        def updatePolicyObservableBox(viewer, action):
            self.policyObservableComboBox.clear()
            policyObservables = OBSERVABLES.getAllObservablesOfType(FunctionOverStateSpaceObservable)
            self.policyObservableComboBox.addItems([policyObservable.title 
                                                for policyObservable in policyObservables])
            if len(policyObservables) > 0:
                self.selectedPolicyObservable = policyObservables[0].title
            else: 
                self.selectedPolicyObservable = None
            
        OBSERVABLES.addObserver(updatePolicyObservableBox)
        
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
        plotWidget.setMinimumSize(600, 500)
        plotWidget.setWindowTitle("Policy")
 
        self.fig = Figure((6.0, 5.0), dpi=100)
        self.axis = self.fig.gca()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(plotWidget)    
        
        # Small text in plot legend
        matplotlib.rcParams.update({'legend.fontsize': 6})
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidget)
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addWidget(self.policyObservableLabel)
        self.vlayout.addWidget(self.policyObservableComboBox)
        self.vlayout.addWidget(self.gridNodesLabel)
        self.vlayout.addWidget(self.gridNodesSlider)
        self.hlayout.addLayout(self.vlayout)
        
        self.setLayout(self.hlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.trajectoryObservableCallback = \
             lambda *transition: self.updateSamples(*transition)
        self.trajectoryObservable.addObserver(self.trajectoryObservableCallback)
        
        self.policyObservable = None
        if self.selectedPolicyObservable:
            self.policyObservable = OBSERVABLES.getObservable(self.selectedPolicyObservable,
                                                              FunctionOverStateSpaceObservable)
            self.policyObservableCallback = \
                 lambda policyEvalFunction: self.updatePolicy(policyEvalFunction)
            self.policyObservable.addObserver(self.policyObservableCallback)
            
    def close(self):
        self.trajectoryObservable.removeObserver(self.trajectoryObservableCallback)
        if self.policyObservable:
            self.policyObservable.removeObserver(self.policyObservableCallback)
        
        super(MountainCarPolicyViewer, self).close()
       
    def updateSamples(self, state, action, reward, succState, episodeTerminated):
        if episodeTerminated:
            self.episodeTerminated = True
                
    def updatePolicy(self, policyEvalFunction):
        # We only update the plot at the end of an episode
        if not self.episodeTerminated: return
        self.episodeTerminated = False
        
        self.lock.acquire()
        
        # Generate 2d state slice
        defaultDimValues = {"position" : 0, "velocity" : 0}    
        stateSlice = generate2dStateSlice(["position", "velocity"], 
                                          self.stateSpace, defaultDimValues,
                                          gridNodesPerDim=self.gridNodesPerDim)
        
        # Compute values that should be plotted
        values, colorMapping = \
                    generate2dPlotArray(policyEvalFunction, stateSlice, 
                                        continuousFunction=False,
                                        shape=(self.gridNodesPerDim, self.gridNodesPerDim)) 
        
        # If all value are None, we cannot draw anything useful
        if values.mask.all():
            self.lock.release()
            return            
        
        # Plot data
        polyCollection = self.axis.pcolor(numpy.linspace(0.0, 1.0, self.gridNodesPerDim), 
                                          numpy.linspace(0.0, 1.0, self.gridNodesPerDim),
                                          values.T)

        # Some dummy code that creates patches that are not shown but allow
        # for a colorbar
        from matplotlib.patches import Rectangle
        linearSegmentedColorbar = polyCollection.get_cmap()
        patches = []
        functionValues = []
        for functionValue, colorValue in colorMapping.items():
            if isinstance(functionValue, tuple):
                functionValue = functionValue[0] # deal with '(action,)'
            normValue = polyCollection.norm(colorValue)
            if isinstance(normValue, numpy.ndarray): 
                normValue = normValue[0] # happens when function is constant
            rgbaColor = linearSegmentedColorbar(normValue)
         
            p = Rectangle((0, 0), 1, 1, fc=rgbaColor)
            functionValues.append(functionValue)
            patches.append(p)
        self.axis.legend(patches, functionValues)
        
        # Labeling etc.
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        self.axis.set_xlabel("position")
        self.axis.set_ylabel("velocity")
        self.axis.legend()
        
        self.canvas.draw()
        
        self.lock.release()
        
    def _policyObservableChanged(self, selectedPolicyObservable):
        self.lock.acquire()
        if self.policyObservable:
            # Disconnect from old policy observable
            self.policyObservable.removeObserver(self.policyObservableCallback)
        
        # Determine new observed policy observable
        self.selectedPolicyObservable = str(selectedPolicyObservable)
        
        # Connect to new policy observable
        self.policyObservable = OBSERVABLES.getObservable(self.selectedPolicyObservable,
                                                          FunctionOverStateSpaceObservable)
        self.policyObservableCallback = \
             lambda policyEvalFunction: self.updatePolicy(policyEvalFunction)
        self.policyObservable.addObserver(self.policyObservableCallback)
        
        self.actions = []
        self.lock.release()
          
    
    def _changeGridNodes(self):
        self.gridNodesPerDim = self.gridNodesSlider.value()
        self.gridNodesLabel.setText("Grid Nodes Per Dimension: %s" 
                                    % self.gridNodesPerDim )

