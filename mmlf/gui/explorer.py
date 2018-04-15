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
import sys
import yaml
from threading import Thread
from copy import deepcopy

from mmlf import QtGui, QtCore
import mmlf
from mmlf.gui.viewers import VIEWERS
from mmlf.framework.observables import OBSERVABLES, StateActionValuesObservable, \
                                    FunctionOverStateSpaceObservable
from mmlf.gui.log_viewer import LogViewer

class MMLFExplorer(QtGui.QWidget):
        
    def __init__(self, parent=None):
        super(MMLFExplorer, self).__init__(parent)
        self.parent = parent
        
        self.world = None
        
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
        
        # The main layout
        self.vlayout = QtGui.QVBoxLayout()
        # Create WorldConfigEditor and load default configuration
        from mmlf.gui.world_config_editor import WorldConfigEditor # must be imported lazily!s
        self.worldConfigEditor = WorldConfigEditor(self, self.vlayout)
        worldConfig = yaml.load(open(mmlf.getRWPath() + os.sep + "config" 
                                     + os.sep + "mountain_car" + os.sep 
                                     + "world_td_lambda_exploration.yaml"))
        # Add missing entries for the monitor to worldConfig
        from mmlf.framework.monitor import Monitor
        for key in Monitor.DEFAULT_CONFIG_DICT.keys():
            if key not in worldConfig["monitor"]:
                worldConfig["monitor"][key] = deepcopy(Monitor.DEFAULT_CONFIG_DICT[key]) 
        self.worldConfigEditor._initFromWorldConfigObject(worldConfig)
        
        # Viewer Selection
        viewerLabel = QtGui.QLabel("Viewer")
        self.viewerComboBox = QtGui.QComboBox(self)
        self.viewerComboBox.addItems(sorted(VIEWERS.getViewerNames()))
        self.viewerComboBox.setToolTip("Viewer that will be added to monitor "
                                       "progress of agent in environment")
        self.selectedViewer = sorted(VIEWERS.getViewerNames())[0]
        viewerAddButton = QtGui.QPushButton("Add")
              
        parent.connect(self.viewerComboBox,
                       QtCore.SIGNAL('activated (const QString&)'), 
                       self._viewerSelectionChanged)
        self.connect(viewerAddButton, QtCore.SIGNAL('clicked()'), 
                     self._addViewer)
        # Automatically update viewer combobox when new viewers are created
        # during runtime
        def updateViewersComboBox(viewer, action):
            self.viewerComboBox.clear()
            self.viewerComboBox.addItems(sorted(VIEWERS.getViewerNames()))
            self.selectedViewer = sorted(VIEWERS.getViewerNames())[0]           
        self.updateViewersComboBox = updateViewersComboBox
        
        VIEWERS.addObserver(self.updateViewersComboBox)

        # World control
        self.worldControlButton = QtGui.QPushButton("Init World")
        self.worldStopButton = QtGui.QPushButton("Stop World")
        self.worldStepButton = QtGui.QPushButton("Step")
        self.worldEpisodeButton = QtGui.QPushButton("Episode")
    
        self.connect(self.worldControlButton, QtCore.SIGNAL('clicked()'), 
                     self._controlWorld)
        self.connect(self.worldStopButton, QtCore.SIGNAL('clicked()'), 
                     self._stopWorld)
        self.connect(self.worldStepButton, QtCore.SIGNAL('clicked()'), 
                     self._executeSteps)
        self.connect(self.worldEpisodeButton, QtCore.SIGNAL('clicked()'), 
                     self._executeEpisode)
        
        # Load and store world config
        configLoadButton = QtGui.QPushButton("Load Config")
        configStoreButton = QtGui.QPushButton("Store Config")
        
        self.connect(configLoadButton, QtCore.SIGNAL('clicked()'), 
                     self._loadConfig)       
        self.connect(configStoreButton, QtCore.SIGNAL('clicked()'), 
                     self.worldConfigEditor.storeConfig)
        
        # Monitor Configuration
        self.monitorConfigureButton = QtGui.QPushButton("Configure Monitor")
        parent.connect(self.monitorConfigureButton, QtCore.SIGNAL('clicked()'), 
                       self._configureMonitor)
        
        # Help button
        helpButton = QtGui.QPushButton("Help")
        parent.connect(helpButton, QtCore.SIGNAL('clicked()'), 
                       self._help)
    
        self.worldStopButton.setEnabled(False)
        self.worldStepButton.setEnabled(False)
        self.worldEpisodeButton.setEnabled(False)
        self.monitorConfigureButton.setEnabled(False)
        
        # Log viewer
        self.mmlfLogView = LogViewer(self)
        # World info viewer
        self.mmlfWorldInfoView = WorldInfoViewer()
        self._updateWorldInfoView()
        self.connect(self.worldConfigEditor.environmentComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._updateWorldInfoView)
        self.connect(self.worldConfigEditor.agentComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     self._updateWorldInfoView)
                        
        # Creating the main Tab-Widget
        self.tabWidget = QtGui.QTabWidget(self)
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.addTab(mmlfStdOutViewer, "StdOut/Err")
        self.tabWidget.addTab(self.mmlfLogView, "MMLF Log")
        self.tabWidget.addTab(self.mmlfWorldInfoView, "World Info")
        self.tabWidget.setCurrentWidget(self.mmlfWorldInfoView)
        self.connect(self.tabWidget, QtCore.SIGNAL('tabCloseRequested (int)'), 
                     self._closeTab)

        # Creating layouts       
        self.hlayoutViewer = QtGui.QHBoxLayout()
        self.hlayoutViewer.addWidget(viewerLabel)
        self.hlayoutViewer.addWidget(self.viewerComboBox)
        self.hlayoutViewer2 = QtGui.QHBoxLayout()
        self.hlayoutViewer2.addWidget(viewerAddButton)
        self.hlayoutViewer.addLayout(self.hlayoutViewer2)
        
        self.hlayoutWorld = QtGui.QHBoxLayout()
        self.hlayoutWorld.addWidget(self.worldControlButton)
        self.hlayoutWorld.addWidget(self.worldStopButton)
        self.hlayoutWorld.addWidget(self.worldStepButton)
        self.hlayoutWorld.addWidget(self.worldEpisodeButton)
        self.hlayoutWorld.addWidget(configLoadButton)
        self.hlayoutWorld.addWidget(configStoreButton)
        self.hlayoutWorld.addWidget(self.monitorConfigureButton)
        self.hlayoutWorld.addWidget(helpButton)
        
        self.vlayout.addLayout(self.hlayoutViewer)
        self.vlayout.addLayout(self.hlayoutWorld)
        self.vlayout.addWidget(self.tabWidget)
        
        self.setLayout(self.vlayout)
    
    def tearDown(self):
        if self.world is not None:
            self._stopWorld()

    def _help(self):
        url = self.parent.getHTMLDocumentationBase() + os.sep + "tutorials" \
                + os.sep +  "quick_start_gui.html#explorer"
        self.parent.documentationTab.load(QtCore.QUrl(url))
        self.parent.tabWidget.setCurrentWidget(self.parent.documentationTab)
                
    def _viewerSelectionChanged(self, selectedViewer):
        self.selectedViewer = str(selectedViewer)
        
    def _addViewer(self):
        viewerWidget = VIEWERS.getViewer(self.selectedViewer)()
        self.tabWidget.addTab(viewerWidget, self.selectedViewer)

    def _closeTab(self, tabIndex):
        if tabIndex <= 2: return # The stdout and log windows cannot be removed
        self.tabWidget.widget(tabIndex).close()
        self.tabWidget.removeTab(tabIndex)
        
    def _updateWorldInfoView(self):
        from mmlf.worlds import MMLF_ENVIRONMENTS
        from mmlf.agents import MMLF_AGENTS
        selectedEnvironment = self.worldConfigEditor.selectedEnvironment
        envInfoString = MMLF_ENVIRONMENTS[selectedEnvironment]["env_class"].__doc__
        self.mmlfWorldInfoView.setEnvironment(selectedEnvironment, envInfoString)
        selectedAgent = self.worldConfigEditor.selectedAgent
        agentInfoString = MMLF_AGENTS[selectedAgent]["agent_class"].__doc__
        self.mmlfWorldInfoView.setAgent(selectedAgent, agentInfoString)
        
    def _loadConfig(self):
        try:
            self.worldConfigEditor.loadConfig()
        except IOError:
            ret = QtGui.QMessageBox.warning(self, "Warning", "No such file.")
            return
        except Exception, e:
            ret = QtGui.QMessageBox.warning(self, "Exception", repr(e))
            return
        self._updateWorldInfoView()
        
    def _configureMonitor(self):
        # Should not be configured before world is loaded
        if self.world is None:
            ret = QtGui.QMessageBox.warning(self, "Warning", 
                                            "Please initialize world first.")
            return
            
        from mmlf.gui.config_editor import ConfigEditorWindow, ListEditorWindow
        from mmlf.framework.monitor import Monitor
            
        # Allow user to configure monitor in GUI 
        def monitorConfiguredCallback(configDict):
            self.worldConfigObject['monitor'] = configDict
            self.worldConfigEditor.monitorConf = configDict
            
        # The non-standard config elements
        nonStandardConfigs = {}
        nonStandardConfigs["stateDims"] \
            = lambda initialValue, closeCallback, parent: \
                ListEditorWindow(self.world.environment.stateSpace.getDimensionNames(),
                                 initialValue, "stateDims", closeCallback,
                                 parent=self)
        nonStandardConfigs["actions"] \
            = lambda initialValue, closeCallback, parent: \
                ListEditorWindow(self.world.environment.actionSpace.getActionList(),
                                 initialValue, "actions", closeCallback,
                                 parent=self)

        observables = []
        for observable in OBSERVABLES.getAllObservablesOfType(FunctionOverStateSpaceObservable):
            observables.append("".join(observable.title.split(" ")[1:]).strip("()"))
        for observable in OBSERVABLES.getAllObservablesOfType(StateActionValuesObservable):
            observables.append("".join(observable.title.split(" ")[1:]).strip("()"))
        observables.append("All")
        nonStandardConfigs["plotObservables"] \
            = lambda initialValue, closeCallback, parent: \
                ListEditorWindow(observables, initialValue, "plotObservables", 
                                 closeCallback, parent=self)
                
        configEditorWindow = \
            ConfigEditorWindow(self.worldConfigObject['monitor'],
                               title = "Monitor",
                               infoString = Monitor.__doc__,
                               closeCallback = monitorConfiguredCallback,
                               nonStandardConfigs=nonStandardConfigs,
                               parent=self)
        configEditorWindow.show()
    
    def _controlWorld(self):
        # Kind of a dispatcher method
        if str(self.worldControlButton.text()) == "Init World":
            self._initWorld()
        elif str(self.worldControlButton.text()) == "Start World":
            self._startWorld()
        elif str(self.worldControlButton.text()) == "Pause World":
            self._pauseWorld()
        elif str(self.worldControlButton.text()) == "Resume World":
            self._resumeWorld()
    
    def _initWorld(self):
        self.worldConfigObject = self.worldConfigEditor.createWorldConfigObject()
        
        if self.world:
            self._stopWorld()
            
        self.world = mmlf.loadWorld(worldConfigObject=self.worldConfigObject,
                                    useGUI=True, 
                                    keepObservers=[self.updateViewersComboBox])
        
        self.worldControlButton.setText("Start World")
        self.worldStopButton.setEnabled(True)
        self.worldStepButton.setEnabled(True)
        self.worldEpisodeButton.setEnabled(True)
        self.monitorConfigureButton.setEnabled(True)
        
        self.tabWidget.setCurrentWidget(self.mmlfLogView)
           
    def _startWorld(self):                   
        # Start the world
        mmlf.log.debug("Starting world...")
        # Run experiment in a separate thread to keep GUI responsive
        def secureWorldRunner():
            try:
                self.world.run(self.worldConfigObject['monitor'])
            except Exception, e:
                errorMessage = e.__class__.__name__ + ": " + str(e)
                self.parent.exceptionOccurredSignal.emit(errorMessage)
                raise
        self.mainThread = Thread(target=lambda : secureWorldRunner(), args=()) 
        self.mainThread.start()
        mmlf.log.debug("Starting world...Done!")
        
        self.worldControlButton.setText("Pause World")
        self.worldStepButton.setEnabled(False)
        self.worldEpisodeButton.setEnabled(False)
        self.monitorConfigureButton.setEnabled(False)
        
    def _pauseWorld(self):
        # Pause the world
        mmlf.log.debug("Pausing world...")
        self.world.iServ.pause = True
            
        self.worldControlButton.setText("Resume World")
        self.worldStepButton.setEnabled(True)
        self.worldEpisodeButton.setEnabled(True)
        
    def _resumeWorld(self):
        # Resume the world
        mmlf.log.debug("Resuming world...")
        self.world.iServ.pause = False
            
        self.worldControlButton.setText("Pause World")
        self.worldStepButton.setEnabled(False)
        self.worldEpisodeButton.setEnabled(False)
        
    def _executeSteps(self):
        try:             
            self.world.executeSteps(n=1, monitorConf=self.worldConfigObject['monitor'])
        except Exception, e:
            errorMessage = e.__class__.__name__ + ": " + str(e)
            self.parent.exceptionOccurredSignal.emit(errorMessage)
            raise

    def _executeEpisode(self):
        try:              
            self.world.executeEpisode(self.worldConfigObject['monitor'])
        except Exception, e:
            errorMessage = e.__class__.__name__ + ": " + str(e)
            self.parent.exceptionOccurredSignal.emit(errorMessage)
            raise
    
    def _stopWorld(self):
        # Stop the world
        mmlf.log.debug("Stopping world...")
        if self.world.iServ:
            self.world.iServ.pause = False
        self.world.stop()
        if hasattr(self, "mainThread"):
            self.mainThread.join()
        mmlf.log.debug("Stopping world... Done!")
        
        self.world = None
        
        self.worldControlButton.setText("Init World")
        self.worldStopButton.setEnabled(False)
        self.worldStepButton.setEnabled(False)
        self.worldEpisodeButton.setEnabled(False)
    
class WorldInfoViewer(QtGui.QWidget):
    
    def __init__(self, parent=None):
        super(WorldInfoViewer, self).__init__(parent)
        
        self.envInfoLabel = QtGui.QLabel("")
        self.envInfoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.envInfoBox = QtGui.QTextBrowser(self)
        
        self.agentInfoLabel = QtGui.QLabel("")
        self.agentInfoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.agentInfoBox = QtGui.QTextBrowser(self)
              
        hlayout = QtGui.QHBoxLayout()
        vlayout1 = QtGui.QVBoxLayout()
        vlayout1.addWidget(self.envInfoLabel)
        vlayout1.addWidget(self.envInfoBox)
        vlayout2 = QtGui.QVBoxLayout()
        vlayout2.addWidget(self.agentInfoLabel)
        vlayout2.addWidget(self.agentInfoBox)
        hlayout.addLayout(vlayout1)
        hlayout.addLayout(vlayout2)
        self.setLayout(hlayout)
        
    def setEnvironment(self, selectedEnvironment, envInfoString):
        self.envInfoLabel.setText("Environment: %s" % selectedEnvironment)
        if envInfoString:
            self.envInfoBox.setText(envInfoString)
        else:
            self.envInfoBox.setText("Documentation missing")

    def setAgent(self, selectedAgent, agentInfoString):
        self.agentInfoLabel.setText("Agent: %s" % selectedAgent)
        if agentInfoString:
            self.agentInfoBox.setText(agentInfoString)
        else:
            self.agentInfoBox.setText("Documentation missing")
        
        