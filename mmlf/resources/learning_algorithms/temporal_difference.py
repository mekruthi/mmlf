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
# Created: 2007/08/16
""" Module for temporal difference learning methods SARSA and Watkins Q.

This module contains the main algorithmic code for the 
temporal difference learning methods SARSA and Watkins Q.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


class TD_Learner(object):
    """ Class that implements general Temporal Difference methods. """
    
    def __init__(self, actions, functionApproximator, featureFct, config):
        #Discount factor
        self.gamma = config['gamma'] 
        #Determines whether the SARSA or Watkins-Q update rule is used
        self.updateRule = config['update_rule']
                
        self.actions = actions
    
        self.featureFct = featureFct
        
        self.functionApproximator = functionApproximator
        
    def getStateValue(self, state):
        """Compute the Q value of *state*.
        
        Compute the Q value of the given state, i.e. the maximum Q value over 
        all actions of the state-action pairs for the given state.
        """
        actionValues = [self.getStateActionValue(state,action)
                           for action in self.actions]
        
        return max(actionValues)
    
    def getStateActionValue(self, state, action):
        """ Compute the Q value for the given *state*-*action* pair """
        features = self.featureFct(state)
        return self.functionApproximator.computeQ(features, action) 
    
    def computeTarget(self, state, action, reward, nextState, nextAction = None):
        """ Compute target Q-value for *state*-*action* pair.
        
        Compute the target Q-value for the state, action pair based on the
        obtained reward and the resulting successor state and the next action 
        taken.
        
        The rule to compute  delta differs between SARSA and Watkins Q
        """
        # Depending on the learning algorithms update rule
        if  nextState == None:
            # If the next state is None, we have reached a terminal state
            # which has value 0
            qNext = 0
        elif self.updateRule == "SARSA" and nextAction != None:
            # For SARSA, the target value is the Q value of the 
            # nextState, nextAction pair 
            qNext = self.getStateActionValue(nextState, nextAction)
        else:
            # For Watkins Q, the target value is the maximal Q value 
            # for nextState among all possible actions
            qNext = self.getStateValue(nextState)
        
        # The target is the reward plus the discount factor times the q value
        # of successor state
        target = reward + self.gamma * qNext 
               
        return target
        
    def train(self, state, action, target):
        """ Modifies the state/action value according to the given delta """
        # Create a training set
        trainingSet = dict()
        
        # Compute feature vector
        features = self.featureFct(state)
        
        # The desired change is the learning rate times the eligibility 
        # times the actual delta 
        trainingSet[(features, action)] = target
         
        # Let the function approximator do the actual change
        self.functionApproximator.train(trainingSet)
    
    def trainOnTraces(self, currentState, currentAction, target, traces):
        """ Train on traces for Q(*currrentState*, *currentAction*)=*target*.
        
        Train using the current *state*-*action* pair, the *target* value
        and by propagating back the changes along the *traces*.
        """
        # Compute the delta for the current state-action pair 
        # and the given target
        delta = target - self.getStateActionValue(currentState,
                                                  currentAction)
        
        # Create a training set
        trainingSet = dict()
        # For all state, action pairs with a significant eligibility 
        for (state, action), eligibility in traces.iteritems():
            currentQ = self.getStateActionValue(state, action)
            
            # Compute feature vector
            features = self.featureFct(state)
            
            # The goal is the current Q value plus delta time the eligibility
            # of state-action pair 
            trainingSet[(features, action)] = currentQ + delta * eligibility
         
        # Let the function approximator do the actual change
        self.functionApproximator.train(trainingSet)
                        
    def trainOnExperience(self, experienceSet):
        """ Train using a *experienceSet* (i.e. (s,a,r,s') quadruples) """
        #Create the trainingSet
        trainingSet = dict()        
        
        oldQValues = []
        newQValues = []
        for i in range(10):
            # For all (s,a,r,s') quadruples
            for experienceTuple in experienceSet:
                state, action, reward, nextState, nextAction = experienceTuple
                
                newQValues.append(self.getStateActionValue(state, action))
                
                # Compute the delta
                target = self.computeTarget(state, action, reward,
                                            nextState, nextAction)
       
                # Compute feature vector
                features = self.featureFct(state)
       
                # Create a training set that stores a mapping from state, action
                # to the actual change for this state            
                trainingSet[(features, action)] = target
                
            bellmannResidual = 0
            if oldQValues != []:
                for oldQ, newQ in zip(oldQValues, newQValues):
                    bellmannResidual += (oldQ - newQ)**2
                
                bellmannResidual = bellmannResidual / len(experienceSet) 
                print "Bellman Residual: %s MaxQ: %s MinQ: %s" % (bellmannResidual,
                                                                  max(newQValues),
                                                                  min(newQValues))
            
            
            oldQValues = newQValues
            newQValues = []
                
            # Let the function approximator do the actual change
            self.functionApproximator.train(trainingSet)
    