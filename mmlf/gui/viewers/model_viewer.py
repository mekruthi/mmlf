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


from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf.gui.viewers.viewer import Viewer

class ModelViewer(Viewer):
    
    def __init__(self, observable, stateSpace):        
        super(ModelViewer, self).__init__()
        
        self.observable = observable
        self.stateSpace = stateSpace
        self.model = None
        
        # Combo Boxes for selecting displaced state space dimensions
        self.dimension1Label = QtGui.QLabel("Dimension X Axis")
        self.comboBox1 = QtGui.QComboBox(self)
        self.comboBox1.addItems(sorted(stateSpace.keys()))
        self.dimension2Label = QtGui.QLabel("Dimension Y Axis")
        self.comboBox2 = QtGui.QComboBox(self)
        self.comboBox2.addItems(sorted(stateSpace.keys()))
        self.comboBox2.setCurrentIndex(1)
        self.dimension1 = sorted(self.stateSpace.keys())[0]
        self.dimension2 = sorted(self.stateSpace.keys())[1]
        self.connect(self.comboBox1, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._dimension1Changed)
        self.connect(self.comboBox2, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._dimension2Changed)
        
        # Combobox for plotting samples versus predictions
        self.plotSamples = False 
        self.foregroundLabel = QtGui.QLabel("Foreground")
        self.comboBoxForeground = QtGui.QComboBox(self)
        self.comboBoxForeground.addItems(["Predictions", "Samples"])
        self.connect(self.comboBoxForeground, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._foregroundChanged)
        
        # Combobox for colouring
        self.colourings = ["Exploration", "Rewards"]
        self.colouring = self.colourings[0]
        self.colouringLabel = QtGui.QLabel("Backgound Colouring")
        self.comboBoxColouring = QtGui.QComboBox(self)
        self.comboBoxColouring.addItems(self.colourings)
        self.connect(self.comboBoxColouring, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._colouringChanged)
        
        # Slider for controlling the resolution of the plots
        self.resolution = 15
        self.resolutionLabel = QtGui.QLabel("Plot resolution: %s" % self.resolution )
        self.resolutionSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.resolutionSlider.setValue(self.resolution)
        self.resolutionSlider.setMinimum(0)
        self.resolutionSlider.setMaximum(50)
        self.resolutionSlider.setTickInterval(5)
        self.resolutionSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        
        self.connect(self.resolutionSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeResolution)
        
        # Slider for controlling the required visits per state
        self.explVisits = 1
        self.explVisitsLabel = QtGui.QLabel("Min visits: %s" % self.explVisits)
        self.explVisitsSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.explVisitsSlider.setValue(self.explVisits)
        self.explVisitsSlider.setMinimum(0)
        self.explVisitsSlider.setMaximum(5)
        self.explVisitsSlider.setTickInterval(1)
        self.explVisitsSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        
        self.connect(self.explVisitsSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeExplVisits)
        
        # Button which causes a redraw
        self.redrawButton = QtGui.QPushButton("Redraw")
        self.connect(self.redrawButton, QtCore.SIGNAL('clicked()'), self._plot)
        
        # Create matplotlib widgets
        plotWidgetModel = QtGui.QWidget(self)
        plotWidgetModel.setMinimumSize(900, 500)
 
        self.figModel = Figure((9.0, 5.0), dpi=100)
        self.axisModel = self.figModel.gca()
        self.canvasModel = FigureCanvas(self.figModel)
        self.canvasModel.setParent(plotWidgetModel)
        
        # Create layout
        self.vlayout = QtGui.QVBoxLayout()
        self.hlayout1 = QtGui.QHBoxLayout()
        self.hlayout1.addWidget(self.dimension1Label)
        self.hlayout1.addWidget(self.comboBox1)
        self.hlayout2 = QtGui.QHBoxLayout()
        self.hlayout2.addWidget(self.dimension2Label)
        self.hlayout2.addWidget(self.comboBox2)
        self.hlayout3 = QtGui.QHBoxLayout()
        self.hlayout3.addWidget(self.foregroundLabel)
        self.hlayout3.addWidget(self.comboBoxForeground)
        self.hlayout4 = QtGui.QHBoxLayout()
        self.hlayout4.addWidget(self.colouringLabel)
        self.hlayout4.addWidget(self.comboBoxColouring)
        self.hlayout5 = QtGui.QHBoxLayout()
        self.hlayout5.addWidget(self.resolutionLabel)
        self.hlayout5.addWidget(self.resolutionSlider)
        self.hlayout6 = QtGui.QHBoxLayout()
        self.hlayout6.addWidget(self.explVisitsLabel)
        self.hlayout6.addWidget(self.explVisitsSlider)
        
        self.vlayout.addLayout(self.hlayout1)
        self.vlayout.addLayout(self.hlayout2)
        self.vlayout.addLayout(self.hlayout3)
        self.vlayout.addLayout(self.hlayout4)
        self.vlayout.addLayout(self.hlayout5)
        self.vlayout.addLayout(self.hlayout6)
        self.vlayout.addWidget(self.redrawButton)
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidgetModel)
        self.hlayout.addLayout(self.vlayout)
        self.setLayout(self.hlayout)
    
    def update(self, model):
        # TODO: Add mutex
        self.model = model
        
    def _plot(self):
        if self.model:
            dimValues = [self.resolution] * self.stateSpace.getNumberOfDimensions()
            self.observable.plot(self.model, self.figModel, self.stateSpace, 
                                 colouring=self.colouring, 
                                 plotSamples=self.plotSamples,
                                 minExplorationValue=self.explVisits,
                                 plotStateDims=[self.dimension1, self.dimension2],
                                 dimValues=dimValues)
            self.canvasModel.draw()
        
    def _dimension1Changed(self, comboBoxIndex):
        self.dimension1 = sorted(self.stateSpace.keys())[comboBoxIndex]
        
    def _dimension2Changed(self, comboBoxIndex):
        self.dimension2 = sorted(self.stateSpace.keys())[comboBoxIndex]
        
    def _foregroundChanged(self, comboBoxIndex):
        self.plotSamples = (comboBoxIndex != 0)
        
    def _colouringChanged(self, comboBoxIndex):
        self.colouring = self.colourings[comboBoxIndex]
        
    def _changeResolution(self):
        self.resolution = self.resolutionSlider.value()
        self.resolutionLabel.setText("Plot resolution: %s" % self.resolution)
        
    def _changeExplVisits(self):
        self.explVisits = self.explVisitsSlider.value()
        self.explVisitsLabel.setText("Min visits: %s" % self.explVisits)
