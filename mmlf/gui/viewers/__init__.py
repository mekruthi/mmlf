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


from mmlf.framework.observables import Observable

class AllViewers(Observable):
    """ A class that collects all viewers and implements the observable interface. """
    def __init__(self):
        super(AllViewers, self).__init__("All viewers")
         
        self.allViewers = dict()
         
    def addViewer(self, viewer, name):
        self.allViewers[name] = viewer
        # Notify all observers by calling them
        for observer in self.observers:
            observer(viewer, "added")
            
    def removeViewer(self, viewer, name):
        self.allViewers.pop(name)
        # Notify all observers by calling them
        for observer in self.observers:
            observer(viewer, "removed")
            
    def getViewerNames(self):
        return self.allViewers.keys()
    
    def getViewer(self, viewerName):
        return self.allViewers[viewerName]

# An object that contains a mapping from viewer name to viewer factory method
VIEWERS = AllViewers()

# Import all standard viewer modules
import mmlf.gui.viewers.float_stream_viewer