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
import yaml

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath("../../"))
    
from copy import deepcopy
    
import mmlf 
from mmlf import QtCore, QtGui
from mmlf.worlds import MMLF_ENVIRONMENTS, MMLF_WORLDS
from mmlf.agents import MMLF_AGENTS
from mmlf.gui.config_editor import ConfigEditorWindow
from mmlf.framework.monitor import Monitor


class WorldConfigEditor(object):
    
    def __init__(self, parent=None, vlayout=None, worldConfigObject=None):
        self.parent = parent
                
        self.monitorConf = Monitor.DEFAULT_CONFIG_DICT
        
        self._environmentSelectionChanged(sorted(MMLF_ENVIRONMENTS.keys())[0])
        self._agentSelectionChanged(sorted(MMLF_AGENTS.keys())[0])   
    
        # Environment Configuration
        environmentLabel = QtGui.QLabel("Environment")
        self.environmentComboBox = QtGui.QComboBox(self.parent)
        self.environmentComboBox.addItems(sorted(MMLF_ENVIRONMENTS.keys()))
        self.environmentComboBox.setToolTip("Environment in which the agent will act")
        environmentConfigureButton = QtGui.QPushButton("Configure")
        
        if parent: # Sometime we use this only for loading ...
            parent.connect(self.environmentComboBox,
                           QtCore.SIGNAL('activated (const QString&)'), 
                           self._environmentSelectionChanged)
            parent.connect(environmentConfigureButton, QtCore.SIGNAL('clicked()'), 
                           self._configureEnvironment)
        
        # Agent Configuration
        agentLabel = QtGui.QLabel("Agent")
        self.agentComboBox = QtGui.QComboBox(self.parent)
        self._updateAvailableAgents(self.selectedEnvironment)
        self.agentComboBox.setToolTip("The learning agent")
        agentConfigureButton = QtGui.QPushButton("Configure")
        
        if parent: # Sometimes we use this only for loading ...
            parent.connect(self.agentComboBox,
                           QtCore.SIGNAL('activated (const QString&)'), 
                           self._agentSelectionChanged)
            parent.connect(agentConfigureButton, QtCore.SIGNAL('clicked()'), 
                         self._configureAgent)       
        
        # If we start with a non-default world configuration
        if worldConfigObject is not None:
            # We overwrite the default config
            self._initFromWorldConfigObject(worldConfigObject)
        
        if vlayout is not None: # Sometimes we use this only for loading ...
            # Adding GUI elements to external vlayout
            self.hlayoutEnvironment = QtGui.QHBoxLayout()
            self.hlayoutEnvironment.addWidget(environmentLabel)
            self.hlayoutEnvironment.addWidget(self.environmentComboBox)
            self.hlayoutEnvironment.addWidget(environmentConfigureButton)
            
            self.hlayoutAgent = QtGui.QHBoxLayout()
            self.hlayoutAgent.addWidget(agentLabel)
            self.hlayoutAgent.addWidget(self.agentComboBox)
            self.hlayoutAgent.addWidget(agentConfigureButton)
            
            vlayout.addLayout(self.hlayoutEnvironment)
            vlayout.addLayout(self.hlayoutAgent)
        
    def createWorldConfigObject(self):
        """ Create MMLF world config object for the chosen agent and environment. """
        # Build the nested worldConfigObject successively        
        envConfig = {'moduleName' : MMLF_ENVIRONMENTS[self.selectedEnvironment]["module_name"],
                     'configDict' : self.envConfigDict}
        
        agentConfig = {'moduleName' : MMLF_AGENTS[self.selectedAgent]["moduleName"],
                       'configDict' : self.agentConfigDict}
        
        worldConfigObject = {'worldPackage' : MMLF_WORLDS[self.selectedEnvironment],
                             'environment': envConfig,
                             'agent' : agentConfig,
                             'monitor' : self.monitorConf}
        return worldConfigObject
           
    def loadConfig(self):
        """ Opens dialog to load an MMLF world from yaml file.
        
        Returns whether a file was selected and loaded.
        """
        # Read world configuration from yaml configuration file
        worldConfFile = \
            str(QtGui.QFileDialog.getOpenFileName(self.parent, 
                                                  "Select a MMLF world config file",
                                                  mmlf.getRWPath() + os.sep + "config",
                                                  "MMLF world config files (*.yaml)"))
        if worldConfFile == "": return False # No file was selected
        worldConfig = yaml.load(open(worldConfFile, 'r'))
        
        # Add missing entries for the monitor to worldConfig
        for key in Monitor.DEFAULT_CONFIG_DICT.keys():
            if key not in worldConfig["monitor"]:
                worldConfig["monitor"][key] = deepcopy(Monitor.DEFAULT_CONFIG_DICT[key]) 
        
        self._initFromWorldConfigObject(worldConfig)
        return True

    def storeConfig(self):
        """ Opens dialog to store an MMLF world to a yaml file.
        
        Returns whether a file was selected and stroed.
        """
        worldConfigObject = self.createWorldConfigObject()
                
        # Store world configuration to yaml configuration file
        worldConfDirectory = mmlf.getRWPath() + os.sep + "config" \
                                + os.sep + worldConfigObject["worldPackage"]
        worldConfFile = \
            str(QtGui.QFileDialog.getSaveFileName(self.parent, 
                                                  "Select a MMLF world config file",
                                                  worldConfDirectory,
                                                  "MMLF world config files (*.yaml)"))
            
        if worldConfFile == "": return False # No file was selected
        
        yaml.dump(dict(worldConfigObject), open(worldConfFile, 'w'),
                  default_flow_style=False)
        
        return True
        
    def _initFromWorldConfigObject(self, worldConfig):       
        # Determine environment name based on module name
        found = False
        for envName, envConf in MMLF_ENVIRONMENTS.iteritems():
            if envConf['module_name'] == worldConfig['environment']['moduleName']:
                self._environmentSelectionChanged(envName)
                found = True
                break
        if not found:
            class UnknownEnvironmentException(Exception): pass
            raise UnknownEnvironmentException("Environment %s unknown!"
                                              % worldConfig['environment']['moduleName'])
            
        # Determine agent name based on module name
        found = False    
        for agentName, agentConf in MMLF_AGENTS.iteritems():
            if agentConf['moduleName'] == worldConfig['agent']['moduleName']:
                self._agentSelectionChanged(agentName)
                found = True
                break
        if not found:
            class UnknownAgentException(Exception): pass
            raise UnknownAgentException("Agent %s unknown!"
                                              % worldConfig['agent']['moduleName'])
            
        # Get agent and environment config dicts
        # NOTE: Must be done after _environmentSelectionChanged and _agentSelectionChanged
        self.envConfigDict = worldConfig['environment']['configDict']
        self.agentConfigDict = worldConfig['agent']['configDict']
        
        # Set combo boxes to selected agent and environment
        self.environmentComboBox.setCurrentIndex(self.environmentComboBox.findText(self.selectedEnvironment))
        self.agentComboBox.setCurrentIndex(self.agentComboBox.findText(self.selectedAgent))
        # Remember monitorConfiguration
        self.monitorConf= worldConfig['monitor']
        
    def _updateAvailableAgents(self, selectedEnvironment):
        self.agentComboBox.clear()
        for agentKey, agentDict in sorted(MMLF_AGENTS.iteritems()):
            if agentDict["agentUsableFct"](selectedEnvironment):
                self.agentComboBox.addItem(agentKey)
                
        selectedAgentIndex = self.agentComboBox.findText(self.selectedAgent)
        if selectedAgentIndex >= 0:
            self.agentComboBox.setCurrentIndex(selectedAgentIndex)
        else:
            self._agentSelectionChanged(sorted(MMLF_AGENTS.keys())[0])
        
    def _environmentSelectionChanged(self, selectedEnvironment):
        self.selectedEnvironment = str(selectedEnvironment)
        self._initEnvironment()
        if hasattr(self, "agentComboBox"):
            self._updateAvailableAgents(self.selectedEnvironment)
        
    def _configureEnvironment(self):
        # Allow user to configure environment in GUI 
        def environmentConfiguredCallback(configDict):
            self.envConfigDict = configDict
                
        configEditorWindow = \
            ConfigEditorWindow(self.envConfigDict,
                               title = self.selectedEnvironment,
                               infoString = MMLF_ENVIRONMENTS[self.selectedEnvironment]["env_class"].__doc__,
                               closeCallback = environmentConfiguredCallback,
                               nonStandardConfigs={},
                               parent=self.parent)
        configEditorWindow.show()
        
    def _initEnvironment(self):
        # Initialize self.envConfigDict
        # Hack: Create ConfigEditorWindow and close it immediately to get the
        # default configuration from the selected environment and store it into
        # self.envConfigDict 
        def environmentConfiguredCallback(configDict):
            self.envConfigDict = configDict
            
        configEditorWindow = \
            ConfigEditorWindow(MMLF_ENVIRONMENTS[self.selectedEnvironment]["env_class"].DEFAULT_CONFIG_DICT,
                               title = self.selectedEnvironment,
                               infoString = MMLF_ENVIRONMENTS[self.selectedEnvironment].__doc__,
                               closeCallback = environmentConfiguredCallback, 
                               nonStandardConfigs={},
                               parent=self.parent)

        configEditorWindow.centralWidget.onClose()
        
    def _agentSelectionChanged(self, selectedAgent):
        self.selectedAgent = str(selectedAgent)
        self._initAgent()
        
    def _configureAgent(self):
        # Allow user to configure agent in GUI 
        def agentConfiguredCallback(configDict):
            self.agentConfigDict = configDict
                
        configEditorWindow = \
            ConfigEditorWindow(self.agentConfigDict,
                               title = self.selectedAgent,
                               infoString = MMLF_AGENTS[self.selectedAgent]["agent_class"].__doc__,
                               closeCallback = agentConfiguredCallback,
                               nonStandardConfigs={},
                               parent=self.parent)
        configEditorWindow.show()
        
    def _initAgent(self):
        # Initialize self.agentConfigDict
        # Hack: Create ConfigEditorWindow and close it immediately to get the
        # default configuration from the selected agent and store it into
        # self.agentConfigDict     
        def agentConfiguredCallback(configDict):
            self.agentConfigDict = configDict
                
        configEditorWindow = \
            ConfigEditorWindow(MMLF_AGENTS[self.selectedAgent]["agent_class"].DEFAULT_CONFIG_DICT,
                               title = self.selectedAgent,
                               infoString = MMLF_AGENTS[self.selectedAgent].__doc__,
                               closeCallback = agentConfiguredCallback,
                               nonStandardConfigs={},
                               parent=self.parent)
        
        configEditorWindow.centralWidget.onClose()
        
        
class WorldConfigEditorWindow(QtGui.QMainWindow):
    
    def __init__(self, onCloseCallback=None, parent=None, worldConfigObject=None):
        super(WorldConfigEditorWindow, self).__init__(parent)
        
        self.onCloseCallback = onCloseCallback
        self.parent = parent
        
        self.setWindowTitle("MMLF World Configuration Editor")
        self.centralWidget = QtGui.QWidget()
        
        self.vlayout = QtGui.QVBoxLayout()
        self.worldConfigEditor = \
                    WorldConfigEditor(self, self.vlayout, worldConfigObject)
        
        self.saveButton = QtGui.QPushButton("Save")
        self.connect(self.saveButton, QtCore.SIGNAL('clicked()'), 
                     self._onClose)
        self.vlayout.addWidget(self.saveButton)
        
        self.centralWidget.setLayout(self.vlayout)
        self.setCentralWidget(self.centralWidget)
        
    def _onClose(self):
        self.onCloseCallback(self.worldConfigEditor.createWorldConfigObject())
        self.close()
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    
    worldConfigEditorWindow = WorldConfigEditorWindow()
    def closeCallback(worldConfigObject):
        worldConfigEditorWindow.worldConfigEditor.storeConfig()
    worldConfigEditorWindow.onCloseCallback = closeCallback
    worldConfigEditorWindow.show()
    
    sys.exit(app.exec_())
    
        