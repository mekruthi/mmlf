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
# Created: 2009/01/16
""" This module defines the Radial Base Function function approximator."""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import math

from mmlf.resources.function_approximators.function_approximator \
                    import FunctionApproximator 
                    
# A function that computes the crossproduct of all given lists
crossProduct = lambda ss, row=[], level=0: \
                    len(ss)>1 \
                    and reduce(lambda x,y:x+y,[crossProduct(ss[1:],row+[i],level+1) for i in ss[0]]) \
                    or [row+[i] for i in ss[0]]                     
                    
class RBF_FA(FunctionApproximator):
    """ The Radial Base Function function approximator
    
    This class implements the function approximator interface by using the RBF
    function approximator. A RBF function approximator is composed of several
    radial base functions, one for each action.
    
    **CONFIG DICT** 
        :learning_rate: : The learning rate used internally in the updates.      
    """
    
    DEFAULT_CONFIG_DICT = {'learning_rate' : 1.0}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, learning_rate, **kwargs):
        super(RBF_FA, self).__init__(stateSpace)
        
        self.stateSpace = stateSpace
        self.actions = actions       
        
        self.rbfs = dict([(action, RBF(stateSpace, learning_rate)) 
                                for action in actions])
    
    def computeQ(self, state, action):
        """
        Computes the Q-value of the given state, action pair
        
        It is assumed that a state is a n-dimensional vector 
        where n is the dimensionality of the state space. Furthmore, the
        states must have been scaled externally so that the value of each dimension
        falls into the bin [0,1]. action must be one of the actions given to the constructor
        """
        # Delegate to the RBF that is responsible for this action
        return self.rbfs[action].computeQ(state)
    
    def train(self, trainingSet):
        """
        Trains the function approximator using the given training set.
        
        trainingSet is a dictionary containing training data where the 
        key is the respective (state, action) pair whose Q-value should
        be updated and the dict-value is this desired Q-value.
        """
        #Split the training set according to the action they correspond to
        trainingSets = dict([(action, dict()) for action in self.actions])
        for (state, action), target in trainingSet.iteritems():
            trainingSets[action][state] = target
        
        # Train each RBF seperately
        for action in self.actions:
            self.rbfs[action].train(trainingSets[action])
        
      
class RBF(object):
    """
    This class implements a single RBF. A RBF is composed of RBF features
    that are placed on a regular grid.
    """
    def __init__(self, stateSpace, learningRate):
        
        self.stateSpace = stateSpace
        self.learningRate = learningRate
        
        self.featuresPerDimension = dict()
        for dimName, dimDescr in stateSpace.iteritems():
            self.featuresPerDimension[dimName] = dimDescr["resolution"]
            
        RBF_Feature.featuresPerDimension = self.featuresPerDimension.values()
        
        self.features = dict()
        
        self.activationRange = 2 # Should be approx. 4.6 * sigma**2 (sigma from RBF_Feature)
    
    def computeQ(self, state):
        """ Compute the Q value of this state """
        # The q value is the sum of all feature's activations
        # times their respective weights
        return sum(feature.getValue(state)
                     for feature in self._getActiveFeatures(state))     
         
    
    def train(self, trainingSet):
        """ Train this RBF with the given training set """
        # For all training samples
        for state, target in trainingSet.iteritems():
            # Compute the delta 
            currentQ = self.computeQ(state)
            delta = target - currentQ
            # Update the weights of all features that are 
            # activated for this state
            for feature in self._getActiveFeatures(state):
                feature.weight += self.learningRate * delta * feature.getActivation(state)
            

    def _getActiveFeatures(self, state):
        """ Returns all features that are activated for this state """
        # Determine for each dimension the coordinates of RBF features
        # that might get activated
        i = 0
        dimCoordinateRanges = dict()
        for dimName, numFeatures in self.featuresPerDimension.iteritems():
            coordinate = int(round(state[i] * numFeatures)) 
            dimCoordinateRanges[dimName] = [float(coord) / numFeatures for coord in range(coordinate - self.activationRange,
                                                                                          coordinate  + self.activationRange + 1)] 
            i += 1
           
        relevantCoordinates = crossProduct(dimCoordinateRanges.values())
        
        # Check which coordinate is actually in the sphere with the
        # activation radius around the given state and activate the 
        # corresponding feature 
        activeFeatures = []
        for coordinate in relevantCoordinates:            
            activeFeatures.append(self.features.setdefault(tuple(coordinate),
                                                           RBF_Feature(coordinate)))
            
        #print map(len, dimCoordinateRanges.values()), len(activeFeatures)  
        return activeFeatures
            
    
#    def _getActiveFeatures(self, state):
#        """ Returns all features that are activated for this state """
#        # Determine for each dimension the coordinates of RBF features
#        # that might get activated
#        i = 0
#        dimCoordinateRanges = dict()
#        for dimName, numFeatures in self.featuresPerDimension.iteritems():
#            minCoordinate = int(math.ceil((state[i] - self.activationRadius) * numFeatures)) 
#            maxCoordinate = int(math.floor((state[i] + self.activationRadius) * numFeatures))
#            dimCoordinateRanges[dimName] = [float(coordinate) / numFeatures 
#                                                for coordinate in range(minCoordinate, 
#                                                                        maxCoordinate + 1)]
#            i += 1
#           
#        relevantCoordinates = crossProduct(dimCoordinateRanges.values())
#        
#        # Check which coordinate is actually in the sphere with the
#        # activation radius around the given state and activate the 
#        # corresponding feature 
#        activeFeatures = []
#        for coordinate in relevantCoordinates:
#            distance = dist(state, coordinate)
#            
#            if distance <= self.activationRadius:
#                activeFeatures.append(self.features.setdefault(tuple(coordinate),
#                                                               RBF_Feature(coordinate)))
#                
#        print map(len, dimCoordinateRanges.values()), len(relevantCoordinates), len(activeFeatures)  
#        return activeFeatures
        
class RBF_Feature(object):
    """
    This class represents a single feature of a RBF. 
    """
    
    # The variance (i.e. the width of  this feature extends)
    variance = 0.5 ** 2
    
    def __init__(self, center):
        self.center = center
        self.weight = 0.0
        
    @staticmethod
    def dist(s1,s2):
        distance = 0.0
        for x1, x2, f in zip(s1,s2, RBF_Feature.featuresPerDimension):
            distance += (f*(x1 - x2))**2
        return math.sqrt(distance)
    
    def getValue(self, state):
        """ 
        Returns the contribution of this feature to the 
        Q value of the given state.
        """
        return self.weight * self.getActivation(state)
    
    def getActivation(self, state):
        """
        Returns the activation of this feature for the given state.
        """
        distance = RBF_Feature.dist(self.center, state)
        return math.exp(-distance/(2 * RBF_Feature.variance))