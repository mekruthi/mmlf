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
import logging
logging.getLogger('').setLevel(logging.DEBUG)


PRINTLEVEL ="info"
FILELEVEL = "debug"

FILEFORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(module)s.%(funcName)s(%(filename)s:%(lineno)d) %(message)s"
PRINTFORMAT = "'%(asctime)s %(name)-20s %(levelname)-8s %(message)s"

consoleHandler = None

def setupConsoleLogging(level=PRINTLEVEL):
    """Set up logging to console"""
    global consoleHandler
    # Remove old handler (if any)
    if consoleHandler is not None:
        logging.getLogger('').removeHandler(consoleHandler)
    # Set up output of logging to console
    consoleHandler = logging.StreamHandler()
    # Set log level for console
    if level.lower() == "debug":
        consoleHandler.setLevel(logging.DEBUG)
    elif level.lower() == "info":
        consoleHandler.setLevel(logging.INFO)
    elif level.lower() == "warning":
        consoleHandler.setLevel(logging.WARNING)
                         
    # set a format which is simpler for console use
    formatter = logging.Formatter(PRINTFORMAT)
    # tell the handler to use this format
    consoleHandler.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(consoleHandler)

def setupFileLogging(logdir, level=FILELEVEL):
    """Set up logging to log file"""     
    # Set up output of logging to file
    logFile = logdir + os.sep + "MMLF.log"
    fileHandler = logging.FileHandler(logFile)
    # Set log level
    if level.lower() == "debug":
        fileHandler.setLevel(logging.DEBUG)
    elif level.lower() == "info":
        fileHandler.setLevel(logging.INFO)
    elif level.lower() == "warning":
        fileHandler.setLevel(logging.WARNING)
                         
    # set a format which is simpler for console use
    formatter = logging.Formatter(FILEFORMAT)
    # tell the handler to use this format
    fileHandler.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(fileHandler)    
    
def getLogger(loggerName):
    "Returns a Logger instance associated with the provided logger name"
    return logging.getLogger(loggerName)


