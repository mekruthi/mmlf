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


import math

from mmlf import QtGui, QtCore
from mmlf import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path  

from mmlf.gui.viewers.viewer import Viewer
from mmlf.framework.observables import OBSERVABLES, TrajectoryObservable

class SPBTrajectoryViewer(Viewer):
    
    def __init__(self):        
        super(SPBTrajectoryViewer, self).__init__()
        
        self.lenpole = 1.0
        
        # Get required observables
        self.trajectoryObservable = \
                OBSERVABLES.getAllObservablesOfType(TrajectoryObservable)[0]
        
        # Create matplotlib widgets
        plotWidget = QtGui.QWidget(self)
        plotWidget.setMinimumSize(600, 500)
        plotWidget.setWindowTitle("SPB Cart Viewer")
 
        self.fig = Figure((6.0, 5.0), dpi=100)
        self.axis = self.fig.gca()
        self.axis.set_xlim((-3.125, 3.125))
        self.axis.set_ylim((-0.5, 5.5))

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(plotWidget)    
        self.canvas.draw()
        
        # Create layout
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addWidget(plotWidget)
        self.setLayout(self.hlayout)
        
        # Connect to observer (has to be the last thing!!)
        self.trajectoryObservableCallback = \
             lambda *transition: self._updateSamples(*transition)
        self.trajectoryObservable.addObserver(self.trajectoryObservableCallback)    
        
    def close(self):
        self.trajectoryObservable.removeObserver(self.trajectoryObservableCallback)
        
        super(SPBTrajectoryViewer, self).close()
               
    def _updateSamples(self, state, action, reward, succState, episodeTerminated):
        cartPosition = succState["cartPosition"]
        poleAngularPosition = succState["poleAngularPosition"]
        
        self.axis.clear()
        # Cart
        cartPath = Path([(cartPosition - 1.0, 0), (cartPosition + 1.0, 0),
                         (cartPosition + 1.0, 0.1), (cartPosition - 1.0, 0.1),
                         (cartPosition - 1.0, 0)])
        cartPatch = PathPatch(cartPath, facecolor=(1.0, 0.0, 0.0),
                              edgecolor=(0.0, 0.0, 0.0))
        self.axis.add_patch(cartPatch)
        # Wheels
        wheelPatch1 = Circle([cartPosition - 0.8, -0.2], 0.2, facecolor='k') 
        wheelPatch2 = Circle([cartPosition + 0.8, -0.2], 0.2, facecolor='k')
        self.axis.add_patch(wheelPatch1)
        self.axis.add_patch(wheelPatch2)        
        # Pole
        sintheta = math.sin(poleAngularPosition)
        costheta = math.cos(poleAngularPosition)
        polePath = Path([(cartPosition - 0.1, 0.1), 
                         (cartPosition - 0.1 + self.lenpole*sintheta, 0.1+self.lenpole*costheta),
                         (cartPosition + 0.1 + self.lenpole*sintheta, 0.1+self.lenpole*costheta), 
                         (cartPosition + 0.1, 0.1),
                         (cartPosition - 0.1, 0.1)])
        polePatch = PathPatch(polePath, facecolor=(0.0, 1.0, 0.0),
                              edgecolor=(0.0, 0.0, 0.0))
        self.axis.add_patch(polePatch)
         
        # Redraw   
        self.canvas.draw()
    

            
            
                    
