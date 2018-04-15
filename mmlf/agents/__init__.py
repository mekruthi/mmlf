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
import re

# The pattern which python modules have to match
modulePattern = re.compile("[a-zA-Z0-9][a-zA-Z0-9_]*.py$")

# The root of the search (should be the agents directory)
root = os.sep.join(__file__.split(os.sep)[:-1])
pardir = os.path.abspath(os.sep.join([root, os.pardir]))

# The global dict of MMLF Agents
MMLF_AGENTS = {}

# Construct the agent search path 
agentSearchPath = [root]
for directory in os.listdir(os.sep.join([pardir, "worlds"])):
    directory = os.sep.join([pardir, "worlds", directory])
    if os.path.isdir(directory) \
       and os.path.exists(os.sep.join([directory, "agents"])): 
        agentSearchPath.append(os.sep.join([directory, "agents"]))

docstring = \
""" Package that contains all available MMLF agents.

A list of all agents:
"""

# search all modules in the agent search path
for directory in agentSearchPath:
    for fileName in os.listdir(directory):
        # Compute the package path for the current directory
        dirPath = directory.split(os.sep)
        # Find the last occurrence of "MMLF" in the path
        mmlfIndex = None
        for index, token in enumerate(dirPath):
            if token == "mmlf": mmlfIndex = index
        dirPath = dirPath[mmlfIndex:]
        packagePath = ".".join(dirPath)
        # Check all files if they are python modules
        if modulePattern.match(fileName):
            # Import the module
            moduleName = fileName.split(".")[0]
            modulePath = packagePath + '.' + moduleName
            try:
                module = __import__(modulePath, {}, {}, ["dummy"])
                # If this module exports MDP nodes
                if hasattr(module, "AgentClass"):
                    def agentUsableFctFactory(directory):
                        # This function returns a function that determines
                        # whether a agent can be used in an environment
                        if directory == root:
                            agentUsableFct = lambda environmentName: True 
                        else:
                            from mmlf.worlds import MMLF_WORLDS
                            agentUsableFct = \
                                lambda environmentName: \
                                    MMLF_WORLDS[environmentName] == directory.split(os.sep)[-2]
                        return agentUsableFct
                    MMLF_AGENTS[module.AgentName] = \
                                {"moduleName": moduleName,
                                 "agent_class": module.AgentClass,
                                 "agentUsableFct": agentUsableFctFactory(directory)}
                    docstring += "\t* Agent '*%s*' from module %s\n\t  implemented in class %s.\n" \
                                         % (module.AgentName, moduleName, module.AgentClass.__name__)
                    if module.AgentClass.__doc__:
                        for line in module.AgentClass.__doc__.split("\n")[2:]:
                            if line.strip() == "": break
                            docstring += "\t\t%s\n" % line
            except Exception, e:
                import traceback
                import logging
                logging.warn("Exception during import of %s: %s\n%s" 
                             % (moduleName, e, traceback.format_exc()))

__doc__ = docstring

# Clean up...
del(re, modulePattern, root, pardir, agentSearchPath, directory, line,
    dirPath, mmlfIndex, index, token, packagePath, moduleName, modulePath, module, 
    fileName, docstring)