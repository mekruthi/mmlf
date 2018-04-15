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


from collections import deque

import numpy

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf.framework.observables import OBSERVABLES, FloatStreamObservable
from mmlf.gui.viewers import VIEWERS
from mmlf.gui.viewers.viewer import Viewer 

class FloatStreamViewer(Viewer):
    
    def __init__(self):
        super(FloatStreamViewer, self).__init__()
               
        # Create matplotlib widget
        plotWidget = QtGui.QWidget(self)
        plotWidget.setMinimumSize(800, 500)
 
        fig = Figure((8.0, 5.0), dpi=100)
        self.canvas = FigureCanvas(fig)
        self.canvas.setParent(plotWidget)
        self.axis = fig.gca()

        # Local container for displayed values
        self.values = deque()
        self.times = deque()
        
        # Combo Box for selecting the observable
        self.comboBox = QtGui.QComboBox(self)
        self.floatStreamObservables = \
                OBSERVABLES.getAllObservablesOfType(FloatStreamObservable)
        self.comboBox.addItems(map(lambda x: "%s" % x.title, 
                                   self.floatStreamObservables))
        self.connect(self.comboBox, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._observableChanged)
        # Automatically update combobox when new float stream observables 
        #  are created during runtime
        def updateComboBox(observable, action):
            self.comboBox.clear()
            self.floatStreamObservables = \
                    OBSERVABLES.getAllObservablesOfType(FloatStreamObservable)
            self.comboBox.addItems(map(lambda x: "%s" % x.title, 
                                       self.floatStreamObservables))
        OBSERVABLES.addObserver(updateComboBox)
        
        # The number of values from the observable that are remembered 
        self.windowSize = 64
        
        # Slider for controlling the window size
        self.windowSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.windowSizeSlider.setValue(numpy.log2(self.windowSize))
        self.windowSizeSlider.setMinimum(0)
        self.windowSizeSlider.setMaximum(15)
        self.windowSizeSlider.setTickInterval(1)
        self.windowSizeSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        
        self.connect(self.windowSizeSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeWindowSize)
        
        self.windowSizeLabel = QtGui.QLabel("WindowSize: %s" % self.windowSize)
        
        # The length of the moving window average 
        self.mwaSize = 10
        
        # Slider for controlling the moving average window
        self.mwaSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.mwaSlider.setValue(self.mwaSize)
        self.mwaSlider.setMinimum(1)
        self.mwaSlider.setMaximum(50)
        self.mwaSlider.setTickInterval(10)
        self.mwaSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        
        self.connect(self.mwaSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeMWA)
        
        self.mwaLabel = QtGui.QLabel("Moving Window Average : %s" % self.mwaSize)
        
        # Create layout
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addWidget(self.comboBox)
        self.vlayout.addWidget(self.windowSizeSlider)
        self.vlayout.addWidget(self.windowSizeLabel)
        self.vlayout.addWidget(self.mwaSlider)
        self.vlayout.addWidget(self.mwaLabel)
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidget)
        self.hlayout.addLayout(self.vlayout)
        
        self.setLayout(self.hlayout)
        
        # Handling connecting to observable
        self.observableCallback = lambda time, value, *args: self.update(time, value)
        if len(self.floatStreamObservables) > 0:
            # Show per default the first observable
            self.observable = self.floatStreamObservables[0] 
            # Connect to observer (has to be the last thing!!)
            self.observable.addObserver(self.observableCallback)
        else:
            self.observable = None
            
    def close(self):
        if self.observable is not None:
            # Remove old observable
            self.observable.removeObserver(self.observableCallback)
        
        super(FloatStreamViewer, self).close()
            
        
    def update(self, time, value):
        self.values.append(value)
        self.times.append(time)
        
        if len(self.values) > self.windowSize:
            self.values.popleft()
            self.times.popleft()
        
        self._redraw()
        
    def _redraw(self):            
        self.axis.clear()
        if len(self.times) == 0: # No data available 
            return
        self.axis.plot(self.times, self.values, 'k')        
        averageValues = []
        for i in range(len(self.values)):
            effectiveMWASize = min(2*i, 2*(len(self.values) - 1 - i), self.mwaSize)
            start = i - effectiveMWASize/2
            end = i + effectiveMWASize/2 + 1
            averageValues.append(float(sum(list(self.values)[start:end])) / (end - start))
            
        self.axis.plot(self.times, averageValues, 'r')
        
        self.axis.set_xlim((min(self.times), max(self.times)))
        
        self.axis.set_xlabel(self.observable.time_dimension_name)
        self.axis.set_ylabel(self.observable.value_name)
        
        self.canvas.draw()
        
    def _observableChanged(self, comboBoxIndex):
        if self.observable is not None:
            # Remove old observable
            self.observable.removeObserver(self.observableCallback)
        # Get new observable and add as listener
        self.observable = self.floatStreamObservables[comboBoxIndex]
        self.observable.addObserver(self.observableCallback)
        # Remove old values
        self.values = deque()
        self.times = deque()
                
    def _changeWindowSize(self):
        self.windowSize = 2**self.windowSizeSlider.value()
        
        while len(self.values) > self.windowSize:
            self.values.popleft()
            self.times.popleft()
        
        self._redraw()
        
        self.windowSizeLabel.setText("WindowSize: %s" % self.windowSize)

    def _changeMWA(self):
        self.mwaSize = self.mwaSlider.value()
        
        self._redraw()
        
        self.mwaLabel.setText("Moving Window Average : %s" % self.mwaSize)

VIEWERS.addViewer(lambda : FloatStreamViewer(), 'FloatStreamViewer')
