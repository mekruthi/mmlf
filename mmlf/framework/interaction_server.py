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
# Created: 2007/07/19
# Modified: 2011-01, Jan Hendrik Metzen

""" InteractionServer that handles the communication between environment and agent. """

__author__ = "Mark Edgington"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import time

import mmlf.framework.protocol

class InteractionServer(object):
    """ InteractionServer that handles the communication between environment and agent.
    
    It controls the interaction between the environment and the agents, and allows
    direct control of various aspects of this interaction.
    """
    
    def __init__(self, world, monitor, initialize=True):        
        # Set world
        self.world = world   
        
        self.monitor = monitor     
        
        # a timestamp for the agent communication
        self.agentCommunicationTimestamp = 0
        
        # if this flag is true the loop will stop before the next iteration
        self.pause = False
        
        # if this flag is true the loop will stop before the next iteration
        self.stopOnNextIteration = False
        
        # if this flag is true this world is started the first time
        # which means the loopInitialize method must be called
        self.initialize = initialize
                
        # Counts the number of episodes the agent has acted in the environment
        self.episodeCounter = 0
    
    def loopInitialize(self):
        """ Inform environment about agent by giving its agentInfo. """
        agentInfo = self.world.agent.agentInfo

        messageObject = mmlf.framework.protocol.GiveAgentInfo(agentInfo = agentInfo)
        self.world.environmentPollMethod(messageObject) # give list to environment

    
    def loopIteration(self):
        """ Perform one iteration of the interaction server loop. 
        
        Call the environmentPollMethod once and based on the result,
        decide if and how to call the agentPollMethod
        """
        cmdObject = self.world.environmentPollMethod(mmlf.framework.protocol.GetWish())

        if cmdObject == None: # then treat it like a no-op
            time.sleep(0) # causes the cpu to change to any other >= priority thread to prevent busy loops
            return self.iterationDebugInfo
        
        # do something based on what the environment said it wants
        messageName = cmdObject["messageName"]
        if messageName in ["setStateSpace", "setState", "setActionSpace",
                           "giveReward"]:
            agentCommandObject = cmdObject # we pass the command on directly to the agent
            self.world.agentPollMethod(agentCommandObject)        
        elif messageName == "getAction":
            agentCommandObject = cmdObject # we pass the command on directly to the agent(s)
            
            # For profiling: Uncomment the following lines and comment out 
            # resultActionObject = self.world.agentPollMethod(agentCommandObject)
            # self.world.environmentPollMethod(resultActionObject)
#            if self.agentCommunicationTimestamp % 1000 == 0:
#                def doCommunication():
#                    resultActionObject = self.world.agentPollMethod(agentCommandObject)
#                    self.world.environmentPollMethod(resultActionObject) # send action back to environment     
#                __import__("cProfile").runctx("doCommunication()", 
#                                              globals(), locals())               
#            else:
#                resultActionObject = self.world.agentPollMethod(agentCommandObject)
#                self.world.environmentPollMethod(resultActionObject) # send action back to environment
            
            resultActionObject = self.world.agentPollMethod(agentCommandObject)
            self.world.environmentPollMethod(resultActionObject) # send action back to environment
            
            self.agentCommunicationTimestamp  += 1            
        elif messageName == "nextEpisodeStarted":
            #One episode has ended --> Increment the episode counter
            self.episodeCounter += 1
            
            # Notify the monitor that an episode has finished
            self.monitor.notifyEndOfEpisode()
                        
            agentCommandObject = cmdObject # we pass the command on directly to the agent
            # we get the response of the agent, and parse it to know if the agent wants to enter
            # extended-testing mode
#            import cProfile; cProfile.runctx("self.world.agentPollMethodDict[agentName](agentCommandObject)", globals(), locals())
            self.world.agentPollMethod(agentCommandObject)
            
    def run(self, numOfEpisodes):
        """ Run an MMLF world for *numOfEpisodes* epsiodes. """
        self.stopOnNextIteration = False
        #if the is not true it is the first start of this world
        if self.initialize:
            self.loopInitialize()
        
        # Uncomment for profiling:
#        def doLoopIterations():
#            for i in range(10000):
#                self.loopIteration()
#        import cProfile; cProfile.runctx("doLoopIterations()", globals(), locals(),
#                                         "profile")
#        __import__("sys").exit(0)
        
        while True:
            while self.pause:
                time.sleep(0.1)
            if self.episodeCounter >= numOfEpisodes or self.stopOnNextIteration:
                break
            self.loopIteration()
            
    def stop(self):
        """ Stop the execution of a world. """
        #set flag that the loop stops after the next iteration
        self.stopOnNextIteration = True
            

