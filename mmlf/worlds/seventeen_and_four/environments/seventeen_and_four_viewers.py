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

""" Module with a viewer for Q functions

This viewer assumes that the state space is one dimensional and that there
is only finite set of actions. This viewer can be used in the seventeen and four
world.
"""

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure

from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.state import State
from mmlf.framework.observables import OBSERVABLES, StateActionValuesObservable

class SeventeenAndFourValuefunctionViewer(Viewer):
    
    def __init__(self, stateSpace):        
        super(SeventeenAndFourValuefunctionViewer, self).__init__()
                
        self.stateSpace = stateSpace
        self.states = stateSpace["count"]["dimensionValues"]
        
        # Combo Box for selecting the observable
        self.comboBox = QtGui.QComboBox(self)
        self.stateActionValuesObservables = \
                OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable)
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
        
        # Create matplotlib widgets
        plotWidgetValueFunction = QtGui.QWidget(self)
        plotWidgetValueFunction.setMinimumSize(800, 500)
 
        self.figValueFunction = Figure((8.0, 5.0), dpi=100)
        #self.figValueFunction.subplots_adjust(left=0.01, bottom=0.04, right=0.99, 
        #                               top= 0.95, wspace=0.05, hspace=0.11)
        self.axisValueFunction = self.figValueFunction.gca()
        self.canvasValueFunction = FigureCanvas(self.figValueFunction)
        self.canvasValueFunction.setParent(plotWidgetValueFunction)
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidgetValueFunction)
        self.hlayout.addWidget(self.comboBox) 
        self.setLayout(self.hlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.stateActionValuesObservableCallback = \
             lambda valueAccessFunction, actions: self.updateValues(valueAccessFunction, actions)
        if len(self.stateActionValuesObservables) > 0:
            # Show per default the first observable
            self.stateActionValuesObservable = self.stateActionValuesObservables[0] 
            plotWidgetValueFunction.setWindowTitle (self.stateActionValuesObservable.title)
            
            self.stateActionValuesObservable.addObserver(self.stateActionValuesObservableCallback)
        else:
            self.stateActionValuesObservable = None  
            
    def close(self):
        if self.stateActionValuesObservable is not None:
            # Remove old observable
            self.stateActionValuesObservable.removeObserver(self.stateActionValuesObservableCallback)
        
        super(SeventeenAndFourValuefunctionViewer, self).close()
    
    def updateValues(self, valueAccessFunction, actions):
        self.axisValueFunction.clear()
        for action in actions:
            actionValues = []
            for state in sorted(self.states):
                actionValues.append(valueAccessFunction(State([state], self.stateSpace.values()), 
                                                        action))
        
            self.axisValueFunction.plot(sorted(self.states),
                                        actionValues, 
                                        label=str(action))
            
        self.axisValueFunction.set_xlabel('Sum of cards')
        self.axisValueFunction.set_ylabel('Value')
        self.axisValueFunction.legend()
        self.canvasValueFunction.draw()
        
    def _observableChanged(self, comboBoxIndex):
        if self.stateActionValuesObservable is not None:
            # Remove old observable
            self.stateActionValuesObservable.removeObserver(self.stateActionValuesObservableCallback)
        # Get new observable and add as listener
        self.stateActionValuesObservable = self.stateActionValuesObservables[comboBoxIndex]
        self.stateActionValuesObservable.addObserver(self.stateActionValuesObservableCallback)
        

