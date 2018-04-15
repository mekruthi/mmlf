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

# Author: Mark Edgington 
# Created: 2007/12/01
# Modified: 2008-01, Thijs Jeffry de Haas
# Modified: 2011-01, Jan Hendrik Metzen
""" Module containing a class that represents a RL world in the MMLF."""

__author__ = "Mark Edgington"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import sys
import os
import imp
import threading
import yaml
    
from numpy import inf

import mmlf
from mmlf import log
from mmlf.framework.interaction_server import InteractionServer
from mmlf.framework.monitor import Monitor
from mmlf.framework.mmlf_logging import setupFileLogging
        
class World(object):
    """ Represents a world consisting of one agent and environment 
    
    The world is created based on a configuration dictionary (worldConfigObject).
    and provides methods for creating agent and environment based on this
    specification and for running the world a given number of steps or epsiodes.
    """
    def __init__(self, worldConfigObject, baseUserDir, useGUI):
        self.worldConfigObject = worldConfigObject
        self.baseUserDir = baseUserDir
        self.iServ = None
        
        # We have a newly loaded world
        self.newWorld = True
        
        # Do the actual loading of the world   
        self.setWorldPackageName(worldConfigObject['worldPackage'])
        
        envConfig = worldConfigObject['environment']
        agentConfig = worldConfigObject['agent']
        
        # Create a unique subdirectory of the world log directory for this 
        # particular configuration file if not already existing
        conf_str = "%s%s%s%s" % (envConfig["moduleName"],
                                 sorted(zip(envConfig["configDict"].keys(),
                                            envConfig["configDict"].values())),
                                 agentConfig["moduleName"],
                                 sorted(zip(agentConfig["configDict"].keys(),
                                            agentConfig["configDict"].values())))
       
        self.baseUserDir.createPath([str(abs(hash(conf_str)))],
                                    refName='conflogdir', baseRef='logbase',
                                    force=True)
        
        # Create unique log directory for this particular run based on current 
        # time and process id
        self.baseUserDir.setTime() # Sets the current time
        process_thread_uid = abs(hash("%s%s" % (threading.current_thread().ident,
                                                os.getpid())))
        self.baseUserDir.createPath(["%s_%s" % (self.baseUserDir.timeString, 
                                                process_thread_uid)],
                                    refName='currentlogdir', baseRef='conflogdir',
                                    force=True)  
        setupFileLogging(self.baseUserDir.getAbsolutePath('currentlogdir'))
        
        # set the environment
        self.loadEnvironment(envConfig, useGUI)
        
        # set the agent
        self.loadAgent(agentConfig, useGUI)
        
    def createMonitor(self, monitorConf=None):
        """ Create monitor based on monitorConf. """
        if monitorConf is not None: 
            self.worldConfigObject['monitor'] = monitorConf
        self.monitor = Monitor(self, self.worldConfigObject['monitor'])
        
        # Store a copy of the world specification in the log directory
        yaml.dump(dict(self.worldConfigObject), 
                  open(self.baseUserDir.getAbsolutePath("conflogdir") 
                       + os.sep + "world.yaml", 'w'),
                  default_flow_style=False)
        
    def loadAgent(self, agentConfig, useGUI):
        """ Create agent based on agent config dict. """
        # We can not set the agent of a running world
        if not self.newWorld:
            raise UserWarning("ERROR (world_interface.addAgentToWorld): Can't "
                              "add an agent to a running world")
        
        #get the agentmodule
        agentModule = self._loadAgentModule(self.worldPackageName, 
                                            agentConfig["moduleName"])
        
        #add the agent to the world
        self.agent = agentModule.AgentClass(config=agentConfig,
                                            baseUserDir=self.baseUserDir,
                                            useGUI=useGUI)
                
    def getAgent(self):
        """ Returns the world's agent"""
        return self.agent
    
    def loadEnvironment(self, envConfig, useGUI):
        """ Create environment based on environment config dict. """
        if not self.newWorld:
            raise UserWarning("ERROR (world_interface.loadEnvironment): Can't "
                              "set the environment of a running world")
        
        #get the environmentModule
        envmodule = self._loadEnvironmentModule(self.worldPackageName, 
                                                envConfig["moduleName"])
                
        #instantiate the environment and set it for the active world
        self.environment = envmodule.EnvironmentClass(config=envConfig,
                                                      baseUserDir=self.baseUserDir,
                                                      useGUI=useGUI)
    
    def getEnvironment(self):
        """ Returns the world's environment"""
        return self.environment
        
    def setWorldPackageName(self, worldPackageName):
        """ Set the name of the python package from which the world should be taken. """
        self.worldPackageName = worldPackageName
        
        # Store/Create relevant pathes for this world in the baseUserDir
        self.baseUserDir.createPath([self.worldPackageName], refName="worldbase", 
                                    baseRef="rwconfig", force=True)
        
        # before we import the world, we need to set the default user-directory based on the specified
        # name, and add a logs directory to it.
        self.baseUserDir.createPath(['logs', self.worldPackageName], refName='logbase', 
                                    force=True) # create/define the base log path        

    def environmentPollMethod(self, commandObject):
        """ Let the environment execute the given command object.
        
        Valid command objects are found in the mmlf.framework.protocol module.
        """      
        # in our simple implementation here, we just create a method with the same name as the 
        # command to handle the command.
        methodName = commandObject.pop("messageName")
        method = getattr(self.environment, methodName)
        
        # take all of the entries in the command object as arguments
        argDict = dict(commandObject)
        
        result = method(**argDict)
        
        return result

    def agentPollMethod(self, commandObject):
        """ Let the agent execute the given command object.
        
        Valid command objects are found in the mmlf.framework.protocol module.
        """
        # TODO: Change such that methods are called directly (more object-oriented!)

        # in our simple implementation here, we just create a method with the same name as the 
        # command to handle the command.
        methodName = commandObject.pop("messageName")
        method = getattr(self.agent, methodName)
        
        # take all of the entries in the command object as arguments
        argDict = dict(commandObject)

        result = method(**argDict)
        
        return result
    
    def run(self, numOfEpisodes=inf, monitorConf=None):
        """ Start the execution of the current world.
        
        Let the world run for *numOfEpisodes* episodes.
        """        
        if self.newWorld:
            self.agentInfo = self.agent.agentInfo
            # Create monitor
            self.createMonitor(monitorConf)
            
            # create an IServ Instance
            # if we have a new world, tell the IServ that he needs to initialize
            self.iServ = InteractionServer(world=self, monitor=self.monitor,
                                           initialize=self.newWorld)
            
            # now our world is not new anymore
            self.newWorld = False
        
        #start the interaction server
        self.iServ.run(numOfEpisodes)

    def executeSteps(self, n=1, monitorConf=None):
        """ Executes n steps of the current world. """
        #if it is the first step of a world
        if self.newWorld:
            self.agentInfo = self.agent.agentInfo
            
            # Create monitor
            self.createMonitor(monitorConf)
            
            # now our world is not new anymore
            self.newWorld = False
        
            #create an IServ Instance
            #if we have a new world, tell the IServ that he needs to initialize
            self.iServ = InteractionServer(world = self, monitor=self.monitor,
                                           initialize=self.newWorld)
            
            self.iServ.loopInitialize()
            
            # Perform 2 loop iterations to exchange relevant information
            self.iServ.loopIteration()
            self.iServ.loopIteration()
        
        # Perform n steps (requires 3 loop iterations each)
        for i in range(n*3):
            self.iServ.loopIteration()
                
    def executeEpisode(self, monitorConf=None):
        """ Executes one episode in the current world. """
        #if it is the first step of a world
        if self.newWorld:
            self.agentInfo = self.agent.agentInfo
            
            # Create monitor
            self.createMonitor(monitorConf)
            
            # now our world is not new anymore
            self.newWorld = False
        
            #create an IServ Instance
            #if we have a new world, tell the IServ that he needs to initialize
            self.iServ = InteractionServer(world = self, monitor=self.monitor, 
                                           initialize=self.newWorld)
            
            self.iServ.loopInitialize()
            
            # Perform 2 loop iterations to exchange relevant information
            self.iServ.loopIteration()
            self.iServ.loopIteration()
        
        currentEpisodeNumber = self.iServ.episodeCounter
        # Perform loop iterations until episode Number increases
        while currentEpisodeNumber == self.iServ.episodeCounter:
            self.iServ.loopIteration()
    
    def stop(self):
        """ Halt the execution of the current world. """
        if self.iServ is not None:
            # stop the IServer Loop
            self.iServ.stop()
        
        # Stop the environment
        self.environment.stop()  
        
        # Remove world-specific viewers
        if mmlf.QtGui:
            from mmlf.gui.viewers import VIEWERS
            for viewerName, viewer in VIEWERS.allViewers.items():
                if viewerName == "FloatStreamViewer": continue
                VIEWERS.removeViewer(viewer, viewerName)
            
    def _loadAgentModule(self, worldname, modulename):
        #create SEARCHPATH
        paths = []
        paths.append(self.baseUserDir.getAbsolutePath("worldbase") + os.sep + "agents" )
        paths.append(self.baseUserDir.getAbsolutePath("ro_area", os.path.join("worlds", worldname, "agents")))
        paths.append(self.baseUserDir.getAbsolutePath("ro_area", "agents"))
        
        # We have to load the module this strange way since imp.load_module
        # causes a reload which can cause strange behaviour since class objects
        # etc. are present multiple time.
        sys_path = list(sys.path) # backup of sys.path 
        sys.path.extend(paths)
        module = __import__(modulename)
        return module
    
    def _loadEnvironmentModule(self, worldname, modulename):
        #create SEARCHPATH
        paths = []
        paths.append(self.baseUserDir.getAbsolutePath("worldbase")+"/environments" )
        paths.append(self.baseUserDir.getAbsolutePath("ro_area", os.path.join("worlds", worldname, "environments")))
        
        #find module
        file ,filename,attr = imp.find_module(modulename, paths)
        
        log.debug("Using %s as environment module" %(filename))
        
        #import module
        # TODO: Import the same way as environments?
        module = imp.load_module(modulename, file, filename, attr)
        return module


