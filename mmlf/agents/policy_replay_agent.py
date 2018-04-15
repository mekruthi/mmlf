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

# Author: Jan Hendrik Metzen (jhm@informatik.uni-bremen.de)
# Created: 2008-06-27
""" Agent for replaying a stored policy

This module defines an agent which loads a stored policy and
follows it without improving it.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import cPickle
import os
import random

import mmlf.framework.protocol
from mmlf.agents.agent_base import AgentBase

class PolicyReplayAgent(AgentBase):
    """ Agent that replays a policy that an other agent has learned previously.
    
    Agent which loads a stored policy and follows it without improving it.
    
    **CONFIG DICT** 
        :plotVF: : If true, the value function is plotted periodically
        :policyPath: : The path from which the pickled policy can be loaded. If a directory, all policies pickled in this directory are replayed in sequence    
    """
    
    DEFAULT_CONFIG_DICT = {"plotVF" : False,
                           "policyPath" : "'/tmp'"}
    
    def __init__(self, *args, **kwargs):
       
        self.agentInfo = mmlf.framework.protocol.AgentInfo(
                            versionNumber="0.3",
                            agentName="PolicyReplayAgent",
                            continuousState = True,
                            continuousAction = True,
                            discreteAction = True,
                            nonEpisodicCapable = True)
    
        # Calls constructor of base class
        super(PolicyReplayAgent, self).__init__(*args, **kwargs)
        
        self.policySet = os.path.isdir(self.configDict["policyPath"])
        
        # Initialize the Q-Plot if desired
        if self.configDict["plotVF"]:
            self._initializeVFPlot()
        
        if self.policySet:
            self.agentLog.info("Replaying a set of policies...")
            self.policySet = True
            
            # Load all policies and store a mapping episode_number -> policy
            path = self.configDict["policyPath"]
            policy_files = os.listdir(self.configDict["policyPath"])
            self.agentLog.info("Loading stored policies....")
            self.policies = [(int(policy_file.split('_')[1]), 
                              cPickle.load(open(path + os.sep + policy_file, 'r')))
                                    for policy_file in policy_files ]
            
            self.agentLog.info("Done!")
            self.policies.sort()
            self.policyIndex = -1
        else:
            # Load the specified policy
            self.agentLog.info("Loading stored policy....")
            file = open(self.configDict["policyPath"], 'r')
            self.policy = cPickle.load(file)
            file.close()
            self.agentLog.info("Done!")
      
    ######################  BEGIN COMMAND-HANDLING METHODS ###############################

    def giveReward(self, reward):
        """ Provides a reward to the agent """
        pass
        
    
    def getAction(self):
        """ Request the next action the agent want to execute """
        action = None
        if not self.policySet: # If we have only a single policy
            # Simply follow this policy 
            action = self.policy.evaluate(self.state)
        else:
            # Check which policy is responsible for this episode
            # This is determined by the suffix of the policy name
            
            # Check if the next policy becomes active
            if self.policyIndex + 1 < len(self.policies) \
              and self.episodeCounter >= self.policies[self.policyIndex + 1][0]:
                self.policyIndex += 1
                
            # If We don't have a responsible policy yet
            if self.policyIndex < 0:
                # Act randomly
                actionDictionary = dict() 
                for actionName in self.actionSpace.iterkeys():
                    if self.actionSpace[actionName]["dimensionType"] == "discrete":
                        actionDictionary[actionName] = \
                            random.choice(self.actionSpace[actionName]["dimensionValues"])
                    else:
                        rangeTuple = random.choice(self.actionSpace[actionName]
                                                                    ["dimensionValues"])
                        actionDictionary[actionName] = \
                            random.uniform(rangeTuple[0], rangeTuple[1])
            else:
                # Else: Apply the responsible policy
                policy = self.policies[self.policyIndex][1]
                action = policy.evaluate(self.state)
                
        # Update Q-Plot if activated:
        if self.configDict["plotVF"]:
            self._updateVFPlot(self.state)
            
        # Create an action dictionary 
        # that maps action dimension to chosen action
        actionDictionary = dict()
        for index, actionName in enumerate(self.actionSpace.iterkeys()):
            actionDictionary[actionName] = action[index]

        super(PolicyReplayAgent, self).getAction()

        return self._generateActionObject(actionDictionary)
    
    def nextEpisodeStarted(self):
        """ Informs the agent that a new episode has started."""
        super(PolicyReplayAgent, self).nextEpisodeStarted()
    
    def _initializeVFPlot(self):
        """ Initializes value function plot that shows the last 100 V(s) values """ 
        import pylab
        pylab.ion()
        
        # This list will at any time contain the last 100 V(s)values
        self.vValueList = [0 for i in range(100)]
                
        # Create the plot and remmeber the line that we will manipulate later on
        self.vLine, = pylab.plot(range(100),self.vValueList, label = "V(s)")        
        pylab.ylabel("V(s)")
        pylab.legend(loc = 2) 
        
    def _updateVFPlot(self, state):
        """ Update the value function plot """
        # Updates the value function plot, appending the 
        # V(s) value of the given state.
         
        import pylab, time
        
        # Remove the oldest V(s) value
        self.vValueList = self.vValueList[1:]
        
        #Remember the current q values for later plots
        self.vValueList.append(self.policy.getStateValue(state))
         
        # Update the plot if we have enough samples
        if len(self.vValueList) == 100:
            self.vLine.set_ydata(self.vValueList)
            # Set range of y axis appropriately
            pylab.gca().set_ylim(min(self.vValueList), max(self.vValueList))

            pylab.draw()
            

AgentClass = PolicyReplayAgent
AgentName = "Policy Replay"

