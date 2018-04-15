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
from operator import itemgetter
from itertools import cycle
from threading import Thread

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas, NavigationToolbar
from matplotlib.figure import Figure

import mmlf
from mmlf.gui.viewers import VIEWERS
from mmlf.gui.viewers.viewer import Viewer
from mmlf.gui.statistical_analysis import StatisticalAnalysisWidget

class ExperimentViewer(Viewer):
    
    def __init__(self, tableModel, parent):
        super(ExperimentViewer, self).__init__()
        
        self.parent = parent
        
        # set the table model
        self.tableModel = tableModel
               
        # Combobox to select metric (FloatStreamObservable) that is shown
        self.metricLabel = QtGui.QLabel("Metric")
        self.metricComboBox = QtGui.QComboBox(self)
        self.metricComboBox.addItems(self.tableModel.metrics)
        self.connect(self.metricComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self.tableModel.metricSelectionChanged)
        # Automatically update metric combobox when new metric observables
        # are created during runtime
        def addMetric(metric):
            self.metricComboBox.addItem(metric)
        self.tableModel.metricsObserver.append(addMetric)
        
        self.plotButton = QtGui.QPushButton("Visualize")
        self.connect(self.plotButton, QtCore.SIGNAL('clicked()'), 
                     self._plotResults)
        
        self.statisticsButton = QtGui.QPushButton("Statistics")
        self.connect(self.statisticsButton, QtCore.SIGNAL('clicked()'), 
                     self._analyzeStatistics)
        
        # create the view
        self.tableView = QtGui.QTableView()
        self.tableView.setModel(self.tableModel)
        self.tableView.setMinimumSize(400, 300) # set the minimum size
        self.tableView.setShowGrid(False) # hide grid

        # Start layouting ...
        self.topLineLayout = QtGui.QHBoxLayout()
        self.topLineLayout.addWidget(self.metricLabel)
        self.topLineLayout.addWidget(self.metricComboBox)
        self.topLineLayout.addWidget(self.plotButton)
        self.topLineLayout.addWidget(self.statisticsButton)

        self.layout = QtGui.QVBoxLayout() 
        self.layout.addLayout(self.topLineLayout)
        self.layout.addWidget(self.tableView) 
        self.setLayout(self.layout)
                    
    def _plotResults(self):
        # Create popup window that shows the results
        self.plotWindow = PlotExperimentWindow(self.tableModel,
                                               parent=self)
        self.plotWindow.show()
        
    def _analyzeStatistics(self):
        # Create statistics analysis widget
        statisticsWindow = StatisticalAnalysisWidget(self.tableModel)
        
        self.parent.tabWidget.addTab(statisticsWindow, "Statistical Analysis")
        self.parent.tabWidget.setCurrentWidget(statisticsWindow) 


class PlotExperimentWindow(QtGui.QMainWindow):
    
    def __init__(self, tableModel, parent):  
        super(PlotExperimentWindow, self).__init__(parent)
            
        self.tableModel = tableModel
        # Colors used for different configurations in plots
        self.colors = cycle(["b", "g", "r", "c", "m", "y", "k"])
        self.colorMapping = defaultdict(lambda : self.colors.next())
        # The length of the moving window average 
        self.mwaSize = 2**0
        # Whether we plot each run separately or only their mean
        self.linePlotTypes = ["Each Run", "Average"]
        self.linePlot = self.linePlotTypes[0]
        
        # The central widget
        self.centralWidget = QtGui.QWidget(self)
        
        # Create matplotlib widget
        self.plotWidget = QtGui.QWidget(self.centralWidget)
        self.plotWidget.setMinimumSize(800, 500)
 
        self.fig = Figure((8.0, 5.0), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.plotWidget)
        self.axis = self.fig.gca()
        
        self.mplToolbar = NavigationToolbar(self.canvas, self.centralWidget)
        
        self.mwaLabel = QtGui.QLabel("Moving Window Average: %s" % self.mwaSize)        
        # Slider for controlling the moving average window
        self.mwaSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.mwaSlider.setValue(0)
        self.mwaSlider.setMinimum(0)
        self.mwaSlider.setMaximum(10)
        self.mwaSlider.setTickInterval(10)
        self.mwaSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        
        self.connect(self.mwaSlider, QtCore.SIGNAL('sliderReleased()'), 
                     self._changeMWA)
        
        self.lineLabel = QtGui.QLabel("Plot of agent: ")
        # Combo Box for selecting the observable
        self.lineComboBox = QtGui.QComboBox(self)
        self.lineComboBox.addItems(["Each Run", "Average"])
        self.connect(self.lineComboBox, QtCore.SIGNAL('currentIndexChanged (int)'), 
                     self._linePlotChanged)
    
        # Add a button for replotting
        self.replotButton = QtGui.QPushButton("Update")
        self.connect(self.replotButton, QtCore.SIGNAL('clicked()'), 
                     self._plot)
        
        # Add a button for saving a plot
        self.saveButton = QtGui.QPushButton("Save")
        self.connect(self.saveButton, QtCore.SIGNAL('clicked()'), 
                     self._save)
        
        # Set layout
        self.topLinelayout = QtGui.QHBoxLayout()
        self.topLinelayout.addWidget(self.mwaLabel)
        self.topLinelayout.addWidget(self.mwaSlider)
        self.topLinelayout.addWidget(self.lineLabel)
        self.topLinelayout.addWidget(self.lineComboBox)
        self.topLinelayout.addWidget(self.replotButton)
        self.topLinelayout.addWidget(self.saveButton)
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addLayout(self.topLinelayout)
        self.vlayout.addWidget(self.plotWidget)
        self.vlayout.addWidget(self.mplToolbar)
        self.centralWidget.setLayout(self.vlayout)
        
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle("Current experiment's results")
        
        # Plot the results once upon creation
        self._plot()
        
    def resizeEvent(self, qResizeEvent):
        self.canvas.setMinimumSize(qResizeEvent.size().width()-50, 
                                   qResizeEvent.size().height()-100)
        self.fig.set_size_inches(qResizeEvent.size().width()/100 -0.5, 
                                 qResizeEvent.size().height()/100 - 1)
        
        self._plot()
        self.plotWidget.repaint()
        self.canvas.draw()
        
    def _plot(self):
        self.axis.clear()
        # Update internal data
        data = self.tableModel.getRunDataForSelectedMetric()
        
        def mwaFilter(inData):
            outData = []
            for i in range(len(inData)):
                start = max(0, i - self.mwaSize/2)
                end = min(len(inData), i + self.mwaSize/2)
                outData.append(float(sum(inData[start:end+1])) / (end+1 - start))
            return outData
        
        if self.linePlot == "Each Run":
            # Do the actual plotting
            plottedWorlds = set()
            for worldName, runNumber in data.keys():
                averageValues = mwaFilter(map(itemgetter(1),
                                              data[(worldName, runNumber)]))
            
                if worldName not in plottedWorlds:  
                    self.axis.plot(map(itemgetter(0), data[(worldName, runNumber)]),
                                   averageValues,
                                   color=self.colorMapping[worldName], 
                                   label=str(worldName))
                    plottedWorlds.add(worldName)
                else:
                    self.axis.plot(map(itemgetter(0), data[(worldName, runNumber)]),
                                   averageValues,
                                   color=self.colorMapping[worldName], 
                                   label="_nolegend_")
        elif self.linePlot == "Average":
            agentAvgData = defaultdict(list)
            agentCounter = defaultdict(list)
            worldsLongestRun = dict()
            for worldName, runNumber in data.keys():
                for i in range(len(data[(worldName, runNumber)])):
                    if i >= len(agentAvgData[worldName]):
                        agentAvgData[worldName].append(data[(worldName, runNumber)][i][1])
                        agentCounter[worldName].append(1.0)
                        worldsLongestRun[worldName] = runNumber
                    else:
                        agentAvgData[worldName][i] += data[(worldName, runNumber)][i][1]
                        agentCounter[worldName][i] += 1.0
            for worldName in agentCounter.keys():
                plotData = []
                for i in range(len(agentAvgData[worldName])):
                    plotData.append(agentAvgData[worldName][i] / agentCounter[worldName][i])
                averagePlotData = mwaFilter(plotData)
                self.axis.plot(map(itemgetter(0), data[(worldName, worldsLongestRun[worldName])]),
                               averagePlotData, color=self.colorMapping[worldName],
                               label=str(worldName))
                    
        self.axis.legend(loc = 'best')
        self.axis.set_xlabel("Episode")
        self.axis.set_ylabel(self.tableModel.selectedMetric)
        #Redraw
        self.canvas.draw()
        
    def _changeMWA(self):
        self.mwaSize = 2**self.mwaSlider.value()
        self.mwaLabel.setText("Moving Window Average: %s" % self.mwaSize)
        # Replot
        self._plot()
        
        
    def _linePlotChanged(self, linePlot):
        self.linePlot = self.linePlotTypes[linePlot]
        # Replot
        self._plot()
        
    def _save(self):
        rootDirectory = \
            self.tableModel.rootDirectory if hasattr(self.tableModel, 
                                                     "rootDirectory") \
                else mmlf.getRWPath()
        graphicFileName = \
            str(QtGui.QFileDialog.getSaveFileName(self,
                                                  "Select a file for the stored graphic",
                                                  rootDirectory,   
                                                  "Plots (*.pdf)"))
        self.fig.savefig(str(graphicFileName), dpi=400)
        
    

VIEWERS.addViewer(lambda : ExperimentViewer(), 'ExperimentViewer') 
