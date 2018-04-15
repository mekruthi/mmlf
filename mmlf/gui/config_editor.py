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


from mmlf import QtCore, QtGui

class MMLFModuleGUIConfig(object):
    
    def __init__(self, initialDict, entryDict, comboBox, configureButton, parent):
        self.configDict = dict(initialDict)
        self.entryDict = entryDict
        if 'name' in self.configDict:
            self.selection = self.configDict.pop('name')
            self.selectionKey = 'name'
        elif 'method' in self.configDict:
            self.selection = self.configDict.pop('method')
            self.selectionKey = 'method'
        else:
            self.selection = self.configDict.pop('type')
            self.selectionKey = 'type'
        self.parent = parent
        
        entries = entryDict.keys()
        entries.remove(self.selection)
        entries = [self.selection] + entries
        comboBox.addItems(entries)
        
        parent.connect(comboBox,
                       QtCore.SIGNAL('activated (const QString&)'), 
                       self.selectionChanged)

        parent.connect(configureButton,
                       QtCore.SIGNAL('clicked()'), 
                       self.configure)
        
        # We have to evaluate the standard window once to get default config dict
        self._evaluateStandardWindow()
    
    def selectionChanged(self, selection):
        self.selection = str(selection)
        # Setting config dict to default for this function approximator
        selectedClass = self.entryDict[self.selection]
        self.configDict = selectedClass.DEFAULT_CONFIG_DICT
        # We have to evaluate the standard window once to get default config dict
        self._evaluateStandardWindow()

    def configure(self):
        self.configEditorWindow.show()
        
    def getConfig(self):
        return self.configDict
    
    def setConfig(self, configDict):
        self.configDict = configDict
        self.configDict[self.selectionKey] = self.selection
        
    def _evaluateStandardWindow(self):
        selectedClass = self.entryDict[self.selection]
        self.configEditorWindow = \
                ConfigEditorWindow(self.configDict, self.selection,
                                   selectedClass.__doc__, self.setConfig, 
                                   nonStandardConfigs = {}, parent=self.parent)
        
        self.configEditorWindow.centralWidget.onClose()
        

class ConfigEditorWidget(QtGui.QWidget):
    
    def __init__(self, initialConfigDict, infoString, closeCallback,
                 nonStandardConfigs={}, parent=None):
        super(ConfigEditorWidget, self).__init__(parent)

        self.infoString = infoString
        self.closeCallback = closeCallback
        self.parent = parent

        self.currentDict = {}
        
        self.vspace = QtGui.QLabel("")
        self.infoLabel = QtGui.QLabel("Info")
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.infoBox = QtGui.QTextBrowser(self)
        if infoString:
            self.infoBox.setText(infoString)
        else:
            self.infoBox.setText("Documentation missing")

        self.closeButton = QtGui.QPushButton("Save")
        self.connect(self.closeButton, QtCore.SIGNAL('clicked()'), 
                     self.onClose)

        self.vlayoutLeft = QtGui.QVBoxLayout()
        self.vlayoutRight = QtGui.QVBoxLayout()
        
        self.fillLayoutFromDict(initialConfigDict, self.currentDict, 
                                infoString, prefix = "", 
                                nonStandardConfigs=nonStandardConfigs)
        
        self.hlayout = QtGui.QHBoxLayout()
        self.hlayout.addLayout(self.vlayoutLeft)
        self.hlayout.addLayout(self.vlayoutRight)
        
        self.vlayout = QtGui.QVBoxLayout()
        self.vlayout.addLayout(self.hlayout)
        self.vlayout.addWidget(self.vspace)
        self.vlayout.addWidget(self.infoLabel)
        self.vlayout.addWidget(self.infoBox)
        self.vlayout.addWidget(self.closeButton)
        
        self.setLayout(self.vlayout)

    def fillLayoutFromDict(self, configDict, currentDict, infoString, prefix = "",
                           nonStandardConfigs = {}):
        for key, value in sorted(configDict.items()):
            self.vlayoutLeft.addWidget(QtGui.QLabel(prefix + str(key)))
            if isinstance(value, dict):
                from mmlf.resources.function_approximators.function_approximator \
                                        import FunctionApproximator
                from mmlf.resources.model.model import Model
                from mmlf.resources.planner.planner import Planner
                from mmlf.resources.policy_search.policy_search \
                                        import PolicySearch
                from mmlf.resources.policies.policy import Policy
                from mmlf.resources.optimization.optimizer import Optimizer
                from mmlf.resources.skill_discovery.skill_discovery \
                                        import SkillDiscovery
                entryDicts = {"function_approximator" : FunctionApproximator.getFunctionApproximatorDict(),
                              "preferences_approximator" : FunctionApproximator.getFunctionApproximatorDict(),
                              "model" : Model.getModelDict(),
                              "planner" : Planner.getPlannerDict(),
                              "policy_search" : PolicySearch.getPolicySearchDict(),
                              "policy" : Policy.getPolicyDict(),
                              "optimizer" : Optimizer.getOptimizerDict(),
                              "skill_discovery" : SkillDiscovery.getSkillDiscoveryDict()}
                if key in entryDicts.keys():
                    comboBox = QtGui.QComboBox(self)
                    configureButton = QtGui.QPushButton("Configure")
                                                           
                    config = MMLFModuleGUIConfig(value, entryDicts[key],
                                                 comboBox, configureButton,
                                                 self)
                    currentDict[key] = (lambda config : lambda : config.getConfig())(config) 
                    
                    comboBox.setToolTip(self._searchTooltip(key, infoString))
                    
                    hlayout = QtGui.QHBoxLayout()
                    hlayout.addWidget(comboBox)
                    hlayout.addWidget(configureButton)
                    self.vlayoutRight.addLayout(hlayout)
                else:
                    self.vlayoutRight.addWidget(QtGui.QLabel(""))
                    someDict = dict()
                    # Recursive call
                    self.fillLayoutFromDict(value, someDict, infoString,
                                            prefix = prefix + "\t",
                                            nonStandardConfigs=nonStandardConfigs)
                    currentDict[key] = someDict
            elif key in nonStandardConfigs.keys(): 
                # this key should be handled differently
                currentDict[key] = value
                configureButton = QtGui.QPushButton("Configure")
                configureButton.setToolTip(self._searchTooltip(key, infoString))
                
                def createFunction(currentDict, key, value):
                    def callbackFunction(config):
                        currentDict[key] = config
                        
                    def configureNonStandardConfig():
                        config = nonStandardConfigs[key](value, callbackFunction, 
                                                         parent=self)
                        config.show()
                    return configureNonStandardConfig
                                                       
                self.connect(configureButton, QtCore.SIGNAL('clicked()'), 
                             createFunction(currentDict, key, value))                
                
                self.vlayoutRight.addWidget(configureButton)
            elif value is True or value is False or value in ["True", "False"]:
                checkbox = QtGui.QCheckBox()
                if value in [True, "True"]:
                    checkbox.setChecked(True)
                checkbox.setToolTip(self._searchTooltip(key, infoString))
                self.vlayoutRight.addWidget(checkbox)
                currentDict[key] = (lambda func: lambda : func())(checkbox.isChecked)
            else:
                lineEdit = QtGui.QLineEdit(str(value))
                lineEdit.setToolTip(self._searchTooltip(key, infoString))
                self.vlayoutRight.addWidget(lineEdit)
                currentDict[key] = (lambda func: lambda : func())(lineEdit.text)
                
    def _searchTooltip(self, key, infoString):
        # Searches tooltip in the infoString. Per conventions, the tooltips
        # have to be written in the syntax ":key: tooltip" where key is the
        # respective key and tooltip is the text that is used as tooltip.
        if infoString is None: return ""
        for line in infoString.splitlines():
            if (":%s:" % key) in line:
                tokens = line.split(" : ")
                if len(tokens) > 1:
                    return tokens[1]
                else:
                    continue
        return ""

    def onClose(self):
        def evaluateFunctions(inputDict):
            returnDict = {}
            for key, value in inputDict.iteritems():
                if isinstance(value, dict):
                    returnDict[key] = evaluateFunctions(value)
                else:
                    if isinstance(value, type(lambda x: x)): 
                        returnDict[key] = value()
                        # Get rid of QT-Strings
                        if isinstance(returnDict[key], QtCore.QString):
                            returnDict[key] = str(returnDict[key])
                        if isinstance(returnDict[key], basestring):
                            # Try to evaluate once (str -> int etc.)
                            try:
                                returnDict[key] = eval(returnDict[key])
                            except NameError, e: 
                                # it is actually a string!
                                pass
                    else: 
                        returnDict[key] = value
            return returnDict

        self.closeCallback(evaluateFunctions(self.currentDict))
        
        self.parent.close()
                                                            
class ConfigEditorWindow(QtGui.QMainWindow):
    
    def __init__(self, configDict, title, infoString, closeCallback, 
                 nonStandardConfigs={}, parent=None):
        super(ConfigEditorWindow, self).__init__(parent)
        
        self.centralWidget = ConfigEditorWidget(configDict, infoString,
                                                closeCallback, 
                                                nonStandardConfigs, self)

        self.setWindowTitle(title)
    
        self.setCentralWidget(self.centralWidget)
        self.resize(800, 600)
        

class ListEditorWindow(QtGui.QMainWindow):
    
    def __init__(self, possibleEntries, initialList, title, closeCallback, 
                 parent=None):
        super(ListEditorWindow, self).__init__(parent)
        
        self.setWindowTitle(title)
        self.closeCallback = closeCallback
        self.parent = parent

        self.currentList = []

        self.closeButton = QtGui.QPushButton("Save")
        self.connect(self.closeButton, QtCore.SIGNAL('clicked()'), 
                     self.onClose)

        self.vlayout = QtGui.QVBoxLayout()
        
        self.checkboxes = []
        for entry in possibleEntries:
            checkbox = QtGui.QCheckBox(str(entry))
            if initialList is not None and entry in initialList:
                checkbox.setChecked(True)
                
            self.checkboxes.append(checkbox)       
            self.vlayout.addWidget(checkbox)
            
        self.closeButton = QtGui.QPushButton("Save")
        self.connect(self.closeButton, QtCore.SIGNAL('clicked()'), 
                     self.onClose)
        self.vlayout.addWidget(self.closeButton)
                
        centralWidget = QtGui.QWidget(self)
        centralWidget.setLayout(self.vlayout)
        self.setCentralWidget(centralWidget)
        

    def onClose(self):
        selections = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                selections.append(str(checkbox.text()))
        self.closeCallback(selections)
        
        self.close()
        
