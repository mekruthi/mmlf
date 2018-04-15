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


import os
from collections import defaultdict
from operator import itemgetter
from threading import Thread, Lock
import numpy

from mmlf import QtGui, QtCore
from mmlf.gui.multiprocessing_observables_utils import ObserverDispatcher

class OnlineExperimentResults(QtCore.QAbstractTableModel):
    """ Data structure that gathers results acquired during an MMLF Experimenter run. """
    
    def __init__(self, observableQueue, updateQueue, maxEpisodes): 
        super(OnlineExperimentResults, self).__init__()
        
        # Global lock for this model
        self.lock = Lock()  
        
        # The attribute is used to chop off runs with too many episodes
        # that were caused due to the asynchronous nature of the MMLF
        self.maxEpisodes = maxEpisodes
                
        self.metrics = []
        self.metricsObserver = []
        
        self.runData = defaultdict(lambda : defaultdict(list))
        self.horizontalHeader = ["Min", "Max", "Mean", "StdDev", "Median", 
                                 "Last", "Max. Time"]
        self.verticalHeader = []
        self.selectedMetric = None
        
        # Create ObserverDispatcher that forwards messages from observables
        # that arrive via the updateQueue to the actual observer function
        self.observerDispatcher = ObserverDispatcher(updateQueue)
        self.observerDispatcher.start()
        
        # A function that runs in a separate thread and connect observable 
        # proxies that arrive via the observableQueue to the callback functions
        # of the observers
        def observableHandler():
            self.running = True
            while self.running:
                # Fetch new observable from observable queue
                try:
                    observable = observableQueue.get(timeout=1.0)
                except: # To avoid blocking at the end of experiment
                    continue
                
                self.lock.acquire()
                observableCallback = \
                    lambda observable: lambda *args: self._update(observable.title, *args)
                
                self.observerDispatcher.connectObserverToObservable(observable,
                                                                    observableCallback(observable))

                # Add this metric as option to the combo box in super class
                if observable.title not in self.metrics:
                    for metricObserver in self.metricsObserver:
                        metricObserver(observable.title)
                    self.metrics.append(observable.title)
                    
                if self.selectedMetric == None:
                    self.selectedMetric = observable.title
                    
                self.lock.release()
                        
        observableHandlerThread = Thread(target=observableHandler, args=())  
        observableHandlerThread.start()
        
    def experimentFinished(self):
        """ Call when experiment is finished to stop data collection threads."""
        self.running = False
        self.observerDispatcher.stop()
        
    def rowCount(self, parent):
        self.lock.acquire()
        rowCount = len(self.runData[self.selectedMetric].keys())
        self.lock.release()
        
        return rowCount 
 
    def columnCount(self, parent): 
        self.lock.acquire()
        columnCount = len(self.horizontalHeader)
        self.lock.release()
        
        return columnCount 
 
    def data(self, index, role):
        self.lock.acquire()
        if not index.isValid(): 
            qVariant = QtCore.QVariant() 
        elif role != QtCore.Qt.DisplayRole: 
            qVariant = QtCore.QVariant() 
        else:
            worldName, runNumber = sorted(self.runData[self.selectedMetric].keys())[index.row()]
            
            selectedData = self.runData[self.selectedMetric][worldName, runNumber]
            if index.column() == 0:
                aggr = lambda x: float(numpy.min(map(itemgetter(1), selectedData)))
            elif index.column() == 1:
                aggr = lambda x: float(numpy.max(map(itemgetter(1), selectedData)))
            elif index.column() == 2:
                aggr = lambda x: float(numpy.mean(map(itemgetter(1), selectedData)))
            elif index.column() == 3:
                aggr = lambda x: float(numpy.std(map(itemgetter(1), selectedData)))
            elif index.column() == 4:
                aggr = lambda x: float(numpy.median(map(itemgetter(1), selectedData)))
            elif index.column() == 5:
                aggr = lambda x: float(map(itemgetter(1), selectedData)[-1])
            elif index.column() == 6:
                aggr = lambda x: float(selectedData[-1][0])  
            qVariant = QtCore.QVariant(aggr(selectedData))
            
        self.lock.release()
            
        return qVariant 

    def headerData(self, index, orientation, role):
        self.lock.acquire()
        
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            qVariant = QtCore.QVariant(self.horizontalHeader[index])
        elif orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            qVariant = QtCore.QVariant(self.verticalHeader[index])
        else:
            qVariant = None
            
        self.lock.release()            
        return qVariant 
                    
    def metricSelectionChanged(self, selectedMetric):
        self.lock.acquire()
        self.selectedMetric = str(selectedMetric)
        self.lock.release()     
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        self.emit(QtCore.SIGNAL("layoutChanged()"))   

    def getRunDataForSelectedMetric(self):
        return self.runData[self.selectedMetric]

    def _update(self, name, time, value, worldName, runNumber):
        self.lock.acquire()
        
        changingLayout = (name == self.selectedMetric) \
                            and (worldName, runNumber) not in self.runData[name]
        if changingLayout:
            self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        else:
            # Check if we have to chop off too long runs
            runLength = len(self.runData[name][(worldName, runNumber)]) 
            if runLength >= self.maxEpisodes:
                self.lock.release()
                return # Ignore sample
        
        self.runData[name][(worldName, runNumber)].append((time, value)) 
        
        if changingLayout:
            self.verticalHeader = \
                ["%s Run %s" % (worldName, runNumber) 
                    for worldName, runNumber in sorted(self.runData[name].keys())]
            self.emit(QtCore.SIGNAL("layoutChanged()"))
            
        if name == self.selectedMetric:
            runIndex = sorted(self.runData[name].keys()).index((worldName,
                                                                runNumber))
            self.emit(QtCore.SIGNAL('dataChanged(const QModelIndex &, const QModelIndex &)'),
                                    self.createIndex(runIndex, 0), 
                                    self.createIndex(runIndex, 6))
            
        self.lock.release()


class OfflineExperimentResults(QtCore.QAbstractTableModel):
    """ Data structure that load results of an MMLF Experimenter run. 
    
    The results are looked up under the root directory *rootDirectory.
    """
    
    def __init__(self, rootDirectory): 
        super(OfflineExperimentResults, self).__init__()
                         
        self.rootDirectory = rootDirectory
                       
        self.metrics = []
        self.metricsObserver = []
        
        self.runData = defaultdict(lambda : defaultdict(list))
        self.horizontalHeader = ["Min", "Max", "Mean", "StdDev", "Median", 
                                 "Last", "Max. Time"]
        self.verticalHeader = []
        self.selectedMetric = None
        
        # Load data
        self._loadExperimentResults(rootDirectory)
        
    def _loadExperimentResults(self, rootDirectory):
        def getImmediateSubdirectories(dir):
            return [os.path.join(dir, name) for name in os.listdir(dir)
                    if os.path.isdir(os.path.join(dir, name))]
        
        for worldDirectory in sorted(filter(os.path.isdir, getImmediateSubdirectories(rootDirectory))):
            worldName = worldDirectory.split(os.sep)[-1]
            for runNumber, runDirectory in enumerate(filter(os.path.isdir, 
                                                            getImmediateSubdirectories(worldDirectory))):
                self.verticalHeader.append("%s Run %s" % (worldName, runNumber))
                for dirpath, dirnames, filenames in os.walk(runDirectory):
                    for filename in filter(lambda s: s.endswith(".fso"), filenames):
                        metricName = filename.split('.')[0]
                        logfile = open(dirpath + os.sep + filename, 'r')
                        for line in logfile.readlines():
                            if line.strip() == "": continue
                            time, sep, value = line.partition('\t')
                            self.runData[metricName][(worldName, runNumber)].append((float(time), float(value)))
                        logfile.close()
                          
        self.metrics = sorted(self.runData.keys())
        if len(self.metrics) > 0:
            self.selectedMetric = self.metrics[0]       
              
    def rowCount(self, parent):
        rowCount = len(self.runData[self.selectedMetric].keys())
        return rowCount 
 
    def columnCount(self, parent): 
        columnCount = len(self.horizontalHeader)
        return columnCount 
 
    def data(self, index, role):
        if not index.isValid(): 
            qVariant = QtCore.QVariant() 
        elif role != QtCore.Qt.DisplayRole: 
            qVariant = QtCore.QVariant() 
        else:
            worldName, runNumber = sorted(self.runData[self.selectedMetric].keys())[index.row()]
            
            selectedData = self.runData[self.selectedMetric][worldName, runNumber]
            if index.column() == 0:
                aggr = lambda x: float(numpy.min(map(itemgetter(1), selectedData)))
            elif index.column() == 1:
                aggr = lambda x: float(numpy.max(map(itemgetter(1), selectedData)))
            elif index.column() == 2:
                aggr = lambda x: float(numpy.mean(map(itemgetter(1), selectedData)))
            elif index.column() == 3:
                aggr = lambda x: float(numpy.std(map(itemgetter(1), selectedData)))
            elif index.column() == 4:
                aggr = lambda x: float(numpy.median(map(itemgetter(1), selectedData)))
            elif index.column() == 5:
                aggr = lambda x: float(map(itemgetter(1), selectedData)[-1])
            elif index.column() == 6:
                aggr = lambda x: float(selectedData[-1][0]) 
            qVariant = QtCore.QVariant(aggr(selectedData))
            
        return qVariant 

    def headerData(self, index, orientation, role):       
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            qVariant = QtCore.QVariant(self.horizontalHeader[index])
        elif orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            qVariant = QtCore.QVariant(self.verticalHeader[index])
        else:
            qVariant = None
                     
        return qVariant 
                    
    def metricSelectionChanged(self, selectedMetric):
        self.selectedMetric = str(selectedMetric)
        self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
        self.emit(QtCore.SIGNAL("layoutChanged()"))   

    def getRunDataForSelectedMetric(self):
        return self.runData[self.selectedMetric]

