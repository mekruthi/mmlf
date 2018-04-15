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


import sys
import os
import yaml
from multiprocessing import cpu_count, Queue, active_children

import mmlf
from mmlf import QtGui, QtCore
from mmlf.gui.log_viewer import LogViewer
from mmlf.framework.experiment import runExperiment

class MMLFExperimenter(QtGui.QWidget):
    
    def __init__(self, parent=None):
        super(MMLFExperimenter, self).__init__(parent)
        self.parent = parent
                
        # Redirect stdout/stderr to viewer (first thing to do)
        mmlfStdOutViewer = QtGui.QTextEdit()
        class OutLog(object):       
            def __init__(self, wrappedOutput):
                self.wrappedOutput = wrappedOutput
            def write(self, m):       
                mmlfStdOutViewer.append(m)
                self.wrappedOutput.write(m)
            def flush(self):
                self.wrappedOutput.flush()
                
        sys.stdout = OutLog(sys.stdout)
        sys.stderr = OutLog(sys.stderr)
                
        # A mapping from world name to world config object
        self.worldNameToConfig = {}
        self.worldCounter = 0
        # A QListWidget that displays all
        self.worldListWidget = QtGui.QListWidget(self)
        self.worldListWidget.setMaximumHeight(80)
        
        self.connect(self.worldListWidget, QtCore.SIGNAL('itemSelectionChanged()'), 
                     self._worldSelectionChanged)
        
        # Help button
        helpButton = QtGui.QPushButton("Help")
        parent.connect(helpButton, QtCore.SIGNAL('clicked()'), 
                       self._help)
                
        # Manage world (i.e. experiments) list
        createWorldButton = QtGui.QPushButton("Create World")
        loadWorldButton = QtGui.QPushButton("Load World")
        editWorldButton = QtGui.QPushButton("Edit World")
        removeWorldButton = QtGui.QPushButton("Remove World")
        self.worldNameEdit = QtGui.QLineEdit("")
        self.worldNameEdit.minimumSizeHint = lambda : QtCore.QSize(100,30) 
        
        self.connect(createWorldButton, QtCore.SIGNAL('clicked()'), 
                     self._createConfig)
        self.connect(loadWorldButton, QtCore.SIGNAL('clicked()'), 
                     self._loadConfig)
        self.connect(editWorldButton, QtCore.SIGNAL('clicked()'), 
                     self._editConfig)
        self.connect(removeWorldButton, QtCore.SIGNAL('clicked()'), 
                     self._removeConfig)
        self.connect(self.worldNameEdit, QtCore.SIGNAL('textChanged (const QString&)'), 
                     self._renameConfig)
                
        # Experiment control
        runsLabel = QtGui.QLabel("Runs")
        self.runsEdit = QtGui.QLineEdit("1")
        episodesLabel = QtGui.QLabel("Episodes")
        self.episodesEdit = QtGui.QLineEdit("inf")
        numParallelProcessesLabel = QtGui.QLabel("Parallel running processes (max: %s)" %
                                                 cpu_count())
        self.numParallelProcessesEdit = QtGui.QLineEdit("1")
        self.numParallelProcessesEdit.setReadOnly(True)        
        self.concurrencyComboBox = QtGui.QComboBox(self)
        self.concurrencyComboBox.addItem("Sequential")
        if sys.platform != 'win32': # Windows only supports sequential execution
            self.concurrencyComboBox.addItem("Concurrent")
        self.concurrencyComboBox.setToolTip("Determines whether world processes "
                                            "are started sequentially or concurrently.")
        self.connect(self.concurrencyComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._concurrencySettingChanged)
        
        
        loadExperimentButton = QtGui.QPushButton("Load Experiment")
        storeExperimentButton = QtGui.QPushButton("Store Experiment")
        startExperimentButton = QtGui.QPushButton("Start Experiment")
        loadResultsButton = QtGui.QPushButton("Load Experiment Results")
        self.connect(loadExperimentButton, QtCore.SIGNAL('clicked()'), 
                     self._loadExperiment)
        self.connect(storeExperimentButton, QtCore.SIGNAL('clicked()'), 
                     self._storeExperiment)
        self.connect(startExperimentButton, QtCore.SIGNAL('clicked()'), 
                     self._startExperiment)
        self.connect(loadResultsButton, QtCore.SIGNAL('clicked()'), 
                     self._loadResults)  
        
        # Log viewer
        mmlfLogView = LogViewer(self)
        
        # Creating the main Tab-Widget
        self.tabWidget = QtGui.QTabWidget(self)
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.addTab(mmlfStdOutViewer, "StdOut/Err")
        self.tabWidget.addTab(mmlfLogView, "MMLF Log")
        self.tabWidget.setCurrentWidget(mmlfLogView)
        self.connect(self.tabWidget, QtCore.SIGNAL('tabCloseRequested (int)'), 
                     self._closeTab)

        # Creating layouts
        self.vlayoutWorlds = QtGui.QVBoxLayout()
        self.vlayoutWorlds.addWidget(self.worldListWidget)
        self.hlayoutWorldButtons = QtGui.QHBoxLayout()
        self.hlayoutWorldButtons.addWidget(createWorldButton)
        self.hlayoutWorldButtons.addWidget(loadWorldButton)
        self.hlayoutWorldButtons.addWidget(editWorldButton)
        self.hlayoutWorldButtons.addWidget(removeWorldButton)
        self.hlayoutWorldButtons.addWidget(self.worldNameEdit)
        self.vlayoutWorlds.addLayout(self.hlayoutWorldButtons)
        
        self.vlayoutExperiment = QtGui.QVBoxLayout()
        self.hlayoutRunsEpisodes = QtGui.QHBoxLayout()
        self.hlayoutRunsEpisodes.addWidget(runsLabel)
        self.hlayoutRunsEpisodes.addWidget(self.runsEdit)
        self.hlayoutRunsEpisodes.addWidget(episodesLabel)
        self.hlayoutRunsEpisodes.addWidget(self.episodesEdit)
        self.vlayoutExperiment.addLayout(self.hlayoutRunsEpisodes)
        self.hlayoutParallelProcesses = QtGui.QHBoxLayout()
        self.hlayoutParallelProcesses.addWidget(numParallelProcessesLabel)
        self.hlayoutParallelProcesses.addWidget(self.numParallelProcessesEdit)
        self.hlayoutParallelProcesses.addWidget(self.concurrencyComboBox)
        self.vlayoutExperiment.addLayout(self.hlayoutParallelProcesses)
        self.hlayoutExperimentButtons = QtGui.QHBoxLayout()
        self.hlayoutExperimentButtons.addWidget(loadExperimentButton)
        self.hlayoutExperimentButtons.addWidget(storeExperimentButton)
        self.hlayoutExperimentButtons.addWidget(startExperimentButton)
        self.hlayoutExperimentButtons.addWidget(loadResultsButton)
        self.hlayoutExperimentButtons.addWidget(helpButton)
        self.vlayoutExperiment.addLayout(self.hlayoutExperimentButtons)
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addLayout(self.vlayoutWorlds)
        self.hlayout.addLayout(self.vlayoutExperiment)
        
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addLayout(self.hlayout)
        self.vlayout.addWidget(self.tabWidget)
        
        self.setLayout(self.vlayout)
        
    def tearDown(self):
        import threading
        for thread in reversed(threading.enumerate()):
            del(thread)
        
        for process in active_children():
            process.terminate()
            del(process)
            
    def _help(self):
        url = self.parent.getHTMLDocumentationBase() + os.sep + "tutorials" \
                + os.sep +  "quick_start_gui.html#experimenter"
        print url
        self.parent.documentationTab.load(QtCore.QUrl(url))
        self.parent.tabWidget.setCurrentWidget(self.parent.documentationTab)

    def _closeTab(self, tabIndex):
        if tabIndex <= 1: return # The stdout and log windows cannot be removed
        self.tabWidget.widget(tabIndex).close()
        self.tabWidget.removeTab(tabIndex)
        
    def _concurrencySettingChanged(self):
        if str(self.concurrencyComboBox.currentText()) == "Concurrent":
            self.numParallelProcessesEdit.setReadOnly(False)
        else:
            self.numParallelProcessesEdit.setText("1")
            self.numParallelProcessesEdit.setReadOnly(True)
    
    def _createConfig(self):
        from mmlf.gui.world_config_editor import WorldConfigEditorWindow # must be imported lazily!
        def onCloseCallback(worldConfigObject):
            self._addWorldConfigToExperiment(worldConfigObject)
        
        self.worldConfigEditorWindow = WorldConfigEditorWindow(onCloseCallback, self)
        self.worldConfigEditorWindow.show()
    
    def _loadConfig(self):
        # Load a world config object from file
        from mmlf.gui.world_config_editor import WorldConfigEditor # must be imported lazily!
        self.worldConfigEditor = WorldConfigEditor()
        try:
            loaded = self.worldConfigEditor.loadConfig()
        except IOError:
            ret = QtGui.QMessageBox.warning(self, "Warning", "No such file.")
            return
        except Exception, e:
            ret = QtGui.QMessageBox.warning(self, "Exception", repr(e))
            return
        if not loaded: return # No file was successfully loaded
        worldConfigObject = self.worldConfigEditor.createWorldConfigObject()
        # Add the loaded world config to the experiment list
        self._addWorldConfigToExperiment(worldConfigObject)
        
    def _editConfig(self):
        from mmlf.gui.world_config_editor import WorldConfigEditorWindow # must be imported lazily!

        listItem = self.worldListWidget.item(self.worldListWidget.currentRow())
        if listItem == None: return
        
        worldName = str(listItem.text())

        def onCloseCallback(worldConfigObject):
            self.worldListWidget.takeItem(self.worldListWidget.currentRow())
            self._addWorldConfigToExperiment(worldConfigObject, worldName)
        
        worldConfigObject = self.worldNameToConfig[worldName]
        
        self.worldConfigEditorWindow = \
                WorldConfigEditorWindow(onCloseCallback, self, worldConfigObject)
        self.worldConfigEditorWindow.show()
        
    def _removeConfig(self):
        # Remove the selected item and the corresponding world
        listItem = self.worldListWidget.takeItem(self.worldListWidget.currentRow())
        if listItem == None: return
        self.worldNameToConfig.pop(str(listItem.text()))
        
    def _renameConfig(self, newWorldName):
        # Change name of world in worldListWidget 
        listItem = self.worldListWidget.item(self.worldListWidget.currentRow())
        if listItem == None: return
        oldName = str(listItem.text())
        newName = str(newWorldName)
        listItem.setText(newName)
        # Change the world name under which the world config object is stored
        self.worldNameToConfig[newName] = self.worldNameToConfig.pop(oldName)
        
    def _addWorldConfigToExperiment(self, worldConfigObject, worldName=None):
        if worldName is None:
            worldName = "World%03d" % self.worldCounter
            self.worldCounter += 1
            
        self.worldNameToConfig[worldName] = worldConfigObject
        self.worldListWidget.addItem(worldName)
        
    def _worldSelectionChanged(self):
        listItem = self.worldListWidget.item(self.worldListWidget.currentRow())
        if listItem is not None:
            self.worldNameEdit.setText(str(listItem.text()))
            
    def _loadResults(self):
        # Request user input of directory from which experiment results should be loaded
        resultDirectory = \
            str(QtGui.QFileDialog.getExistingDirectory(self, "Open Directory",
                                                       mmlf.getRWPath()))
        if resultDirectory == "": return # No results selected
        
        # Create performance metric model        
        from mmlf.gui.models.experiment_results import OfflineExperimentResults
        offlineExperimentResults = OfflineExperimentResults(resultDirectory) 
                
        # Create special experiment viewer
        from mmlf.gui.viewers.experiment_viewer import ExperimentViewer
        experimentViewer = ExperimentViewer(offlineExperimentResults, self)
        self.tabWidget.addTab(experimentViewer, "Experiment Statistics")
        self.tabWidget.setCurrentWidget(experimentViewer)
        
    def _getExperimentMetaConfig(self):
        return {"runsPerWorld": str(self.runsEdit.text()), 
                "episodesPerRun": str(self.episodesEdit.text()), 
                "parallelProcesses": str(self.numParallelProcessesEdit.text()),
                "concurrency" : str(self.concurrencyComboBox.currentText())}
        
    def _loadExperiment(self):
        # Request user input of directory from which experiment should be loaded
        experimentDir = \
            str(QtGui.QFileDialog.getExistingDirectory(self, "Open Directory",
                                                       mmlf.getRWPath()))
        if experimentDir == "": return # No experiment was selected
        
        # Load experiment conf
        experimentConf = mmlf.loadExperimentConfig(experimentDir)
        
        # Store how many runs, episodes, and cores can be used
        self.runsEdit.setText(str(experimentConf["runsPerWorld"]))
        self.episodesEdit.setText(str(experimentConf["episodesPerRun"]))
        self.numParallelProcessesEdit.setText(str(experimentConf["parallelProcesses"]))
        self.concurrencyComboBox.setCurrentIndex(
                 self.concurrencyComboBox.findText(str(experimentConf["concurrency"])))
            
        # Add all worlds of the loaded experiment to the world list of GUI
        for worldName, worldConfigObject in experimentConf["worlds"].items():
            self._addWorldConfigToExperiment(worldConfigObject, worldName)
        
    def _storeExperiment(self):
        # Request user input of directory to which experiment should be saved
        experimentDir = \
            str(QtGui.QFileDialog.getExistingDirectory(self, "Open Directory",
                                                       mmlf.getRWPath()))
        if experimentDir == "": return # No experiment was selected
        
        # Store how many runs, episodes, and cores can be used
        experimentConfigFile = experimentDir + os.sep + "experiment_config.yaml"
        yaml.dump(self._getExperimentMetaConfig(), 
                  open(experimentConfigFile, 'w'),
                  default_flow_style=False)
        
        # Store each world in a separate .yaml file
        worldDir = experimentDir + os.sep + "worlds"
        os.mkdir(worldDir)
        for worldName, worldConfigObject in self.worldNameToConfig.iteritems():
            fileName = worldDir + os.sep + worldName + ".yaml"
            yaml.dump(dict(worldConfigObject), open(fileName, 'w'),
                      default_flow_style=False)
                       
    def _startExperiment(self):
        if sys.platform == "win32":
            if not os.path.exists(sys.argv[0] + ".py"):
                import shutil
                shutil.copyfile(sys.argv[0], sys.argv[0] + ".py")       
        # Create experiment config
        experimentConfig = self._getExperimentMetaConfig()
        experimentConfig["worlds"] = self.worldNameToConfig 
        
        # Creating queues for inter-process communication
        observableQueue = Queue()
        updateQueue = Queue()
        
        # Create performance metric model        
        from mmlf.gui.models.experiment_results import OnlineExperimentResults
        from numpy import inf
        onlineExperimentResults = \
            OnlineExperimentResults(observableQueue, updateQueue,
                                    maxEpisodes=eval(experimentConfig["episodesPerRun"])) 
                
        # Create special experiment viewer
        from mmlf.gui.viewers.experiment_viewer import ExperimentViewer
        experimentViewer = ExperimentViewer(onlineExperimentResults, self)
        self.tabWidget.addTab(experimentViewer, "Experiment Statistics")  
        
        # Run experiment in a separate thread to keep GUI responsive
        from threading import Thread
        def experimentThreadFct(experimentConfig, observableQueue,
                                updateQueue, exceptionOccurredSignal):
            runExperiment(experimentConfig, observableQueue,
                          updateQueue, exceptionOccurredSignal)
            onlineExperimentResults.experimentFinished()
        
        launcherThread = Thread(target=experimentThreadFct, 
                                args=(experimentConfig, observableQueue,
                                      updateQueue, self.parent.exceptionOccurredSignal),
                                name="experimentThreadFct")  
        launcherThread.start()        
        