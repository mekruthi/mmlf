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
import scipy.stats
from numpy import mean, median, max, min, nan

from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf import QtGui, QtCore
            
class StatisticalAnalysisWidget(QtGui.QWidget):
    
    def __init__(self, experimentResults, parent=None):
        super(StatisticalAnalysisWidget, self).__init__(parent)
        
        self.experimentResults = experimentResults
        
        # Statistical test
        self.TESTS = {'MannWhitney U-Test': lambda x, y: scipy.stats.mannwhitneyu(x,y)[1],
                      'Student t-test': lambda x, y: scipy.stats.ttest_ind(x,y)[1]/2}
                
        # Create combobox for selecting the metric
        metricsLabel = QtGui.QLabel("Metric")
        self.metricsComboBox = QtGui.QComboBox(self)
        self.metricsComboBox.addItems(self.experimentResults.metrics)
        
        # Text field for the aggregation function
        aggregationLabel = QtGui.QLabel("Aggregation") 
        self.aggregationFctEdit = QtGui.QLineEdit("lambda x: mean(x[:])")
        self.aggregationFctEdit.minimumSizeHint = lambda : QtCore.QSize(100,30)
        self.aggregationFctEdit.setToolTip("Function which maps a time series "
                                           "onto a single scalar value, which "
                                           "is then used as a sample in "
                                           "the statistical hypothesis testing."
                                           "The functions min, max, mean, and "
                                           "median may be used.")
        
        # Create combobox for selecting the test
        testLabel = QtGui.QLabel("Hypothesis test")
        self.testComboBox = QtGui.QComboBox(self)
        self.testComboBox.addItems(self.TESTS.keys()) 
        
        # Text field for the p-Value
        pValueLabel = QtGui.QLabel("p <") 
        self.pValueEdit = QtGui.QLineEdit("0.05")
        self.pValueEdit.minimumSizeHint = lambda : QtCore.QSize(100,30)
        self.pValueEdit.setToolTip("Significance level: The minimal p-Value "
                                   "which is required for something to be "
                                   "considered as significant.")
        
        # button for redoing the statistics for the current setting
        self.updateButton = QtGui.QPushButton("Update")
        self.connect(self.updateButton, QtCore.SIGNAL('clicked()'), 
                     self._analyze)
                
        # Create matplotlib widget
        plotWidget = QtGui.QWidget(self)
        plotWidget.setMinimumSize(500, 500)
 
        fig = Figure((5.0, 5.0), dpi=100)
        fig.subplots_adjust(0.2)
        self.canvas = FigureCanvas(fig)
        self.canvas.setParent(plotWidget)
        self.axis = fig.gca()
        
        # The table for statistics results
        self.significanceTable = QtGui.QTableWidget(self)
        
        # Do the analyzing once for the default values
        self._analyze()
        
        # Create layout
        layout = QtGui.QVBoxLayout()
        hlayout1 = QtGui.QHBoxLayout()
        hlayout1.addWidget(metricsLabel)
        hlayout1.addWidget(self.metricsComboBox)
        hlayout1.addWidget(aggregationLabel)
        hlayout1.addWidget(self.aggregationFctEdit)
        hlayout1.addWidget(testLabel)
        hlayout1.addWidget(self.testComboBox)
        hlayout1.addWidget(pValueLabel)
        hlayout1.addWidget(self.pValueEdit)
        hlayout1.addWidget(self.updateButton)
        hlayout2 = QtGui.QHBoxLayout()
        hlayout2.addWidget(plotWidget)
        hlayout2.addWidget(self.significanceTable)
        layout.addLayout(hlayout1)
        layout.addLayout(hlayout2)
        self.setLayout(layout)
        
    def _analyze(self):
        # get the raw metric's values
        data = self.experimentResults.runData[str(self.metricsComboBox.currentText())]
        # Compute averages over relevant area and sort them according to configuration
        performances = defaultdict(list)
        for (config, run), values in data.iteritems():
            aggregationFct = eval(str(self.aggregationFctEdit.text()))
            performances[config].append(aggregationFct(map(itemgetter(1), values)))
        
        # Do the plotting
        self.axis.clear()
        self.axis.boxplot([performances[key] for key in sorted(performances.keys())])
        self.axis.set_xticklabels(sorted(performances.keys()))
        self.axis.set_ylabel(str(self.metricsComboBox.currentText()))
        
        # Prepare significanceTable
        self.significanceTable.clear()
        self.significanceTable.setRowCount(len(performances.keys()))
        self.significanceTable.setColumnCount(len(performances.keys()))

        # Setting tables headers    
        self.significanceTable.setHorizontalHeaderLabels(
                                    ["y=%s" % key for key in sorted(performances.keys())])
        self.significanceTable.setVerticalHeaderLabels(
                                    ["x=%s" % key for key in sorted(performances.keys())])
        
        # Add actual p-Value into table
        for index1, config1 in enumerate(sorted(performances.keys())):
            for index2, config2 in enumerate(sorted(performances.keys())):
                # Compute p-Value with selected test
                testFct = self.TESTS[str(self.testComboBox.currentText())]
                try:
                    pValue = testFct(performances[config1], performances[config2])
                except ValueError:
                    pValue = nan
                                        
                # Distinguish a>b and a<b
                if mean(performances[config1]) <  mean(performances[config2]):
                    pValue = 1 - pValue
                
                tableWidgetItem = QtGui.QTableWidgetItem("x>y: p = %.4f" % pValue)
                if pValue < float(str(self.pValueEdit.text())):
                    tableWidgetItem.setFont(QtGui.QFont("Times", 10, QtGui.QFont.Bold))
                else:
                    tableWidgetItem.setFont(QtGui.QFont("Times", 10))
                    
                self.significanceTable.setItem(index1, index2, tableWidgetItem)
                
        self.significanceTable.update()
        self.canvas.draw()
                
        
        