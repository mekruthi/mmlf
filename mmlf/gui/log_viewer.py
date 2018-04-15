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


import time
import logging

from mmlf import QtGui, QtCore

class QtStreamHandler(logging.Handler):
    
    def __init__(self, parent,  main):
        logging.Handler.__init__(self)
        self.parent = parent
        self.main = main
        
        self.textWidget = parent
        self.formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
        
        self.buffer = ""
        self.lastUpdateTime = time.time()
       
    def emit(self, record):
        self.buffer += self.formatter.format(record) + "\n"
        if time.time() - self.lastUpdateTime > 0.1:
            self.textWidget.insertPlainText(self.buffer)
            self.textWidget.moveCursor(QtGui.QTextCursor.End)
            
            self.buffer = ""
            self.lastUpdateTime = time.time() 
            
class LogViewer(QtGui.QWidget):
    
    def __init__(self, parent=None):
        super(LogViewer, self).__init__(parent)
        
        # Log levels
        LEVELS = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}
        
        # The actual logger object
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        
        # Create combobox for selecting the log level
        logLevelLabel = QtGui.QLabel("Log level")
        logLevelComboBox = QtGui.QComboBox(self)
        logLevelComboBox.addItems(["DEBUG", "INFO", "WARNING", 'ERROR', 'CRITICAL'])
              
        def updateLogLevel(level):
            self.logger.setLevel(LEVELS[str(level)])            
        
        self.connect(logLevelComboBox,
                     QtCore.SIGNAL('activated (const QString&)'), 
                     updateLogLevel)
 
        # Create log text field
        logTextField = QtGui.QTextEdit()
        logHandler = QtStreamHandler(logTextField,  self)
        self.logger.addHandler(logHandler)
        
        # Create layout
        layout = QtGui.QVBoxLayout()
        hlayout = QtGui.QHBoxLayout()
        hlayout.addWidget(logLevelLabel)
        hlayout.addWidget(logLevelComboBox)
        layout.addLayout(hlayout)
        layout.addWidget(logTextField)
        self.setLayout(layout)
        
    