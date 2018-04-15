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

""" A set of example (state, successor state, reward) transitions
    
Module that can collect a number of most representative example transitions
observed in a domain. If the maximal number of transitions is exceeded, old
transitions from densely populated regions of the state-action space are
removed.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
import copy
import numpy

if "sphinx" not in __import__("sys").argv[0]: # building documentation
    import scikits.ann as ann

from mmlf.resources.model.model import ModelNotInitialized
from mmlf.framework.state import State

class ExampleSet(object):
    """ A set of example (state, successor state, reward) transitions
    
    Objects of this class store state-reward-successor_state transitions.This is
    required in several different places in the model classes.
     
     * *size* The maximal size of the example set. When the example set is full
              old example have to be removed or no new examples can be added.
    """
    def __init__(self, size):        
        # the maximal size of the data set
        self.size = size
        
        # The variables in which the actual model is stored
        self.states = None
        self.succStates = None
        self.rewards = None
        
        # Remember state dimensions to make sure that they are consistent
        self.stateDimensions = None

    def isEmpty(self):
        """ Returns whether this example set is empty """
        return self.states == None or self.states.shape[1] == 0

    def isFull(self):
        """ Returns whether this example set is full """        
        return self.states != None and self.states.shape[1] >= self.size

    def addTransition(self, state, succState, reward):
        """ Add the given transition (state, succState, reward) to the example set.
        
        Return which state transition (if any) has been removed.
         """
        # Lazy initialization when dimensionality is known
        if self.states == None:
            self.states = numpy.zeros((len(state), 0))
            self.succStates = numpy.zeros((len(state), 0))
            self.rewards = numpy.zeros((1, 0))
            self.stateDimensions = state.dimensions
        else:
            assert (self.stateDimensions == state.dimensions)
       
        removedState = None
       
        if not self.isFull() or not hasattr(self, "kdtree"): # No way to remove states
            # Add sample to internal memory
            self.states = numpy.hstack((self.states,
                                        numpy.array([state]).T))
            self.succStates = numpy.hstack((self.succStates,
                                            numpy.array([succState]).T))
            self.rewards = numpy.hstack((self.rewards,
                                         numpy.array([[reward]])))
        else:
            # The example set is full, we remove the nearest neighbor of the added
            # state 
            # Determine the distance of the current example to its 
            # nearest neighbor               
            minDist = self.kdTree.knn(state, 1)[1][0,0]
            
            # Since it is too expensive to compute the closest pair of the 
            # whole example set, we randomly pick some old examples, compute
            # their distance to their respective nearest neighbors and
            # replace the example with the minimal distance.
            replaceIndex = None
            for i in range(25):
                rndIndex = random.randint(0, self.states.shape[1] - 1)
                dist = self.kdTree.knn(self.states.T[rndIndex],
                                       2)[1][0,1]
                if dist < minDist:
                    minDist = dist
                    replaceIndex = rndIndex
            
            # If all old example have a distance larger than the new example,
            # we ignore the new example
            if replaceIndex == None:
                return None

            # Remember which state transition has been removed and return that 
            # at the end of the method
            removedState = copy.copy(self.states.T[replaceIndex])
            
            # Replace the nearest neighbor by the current state
            self.states.T[replaceIndex] = state
            self.succStates.T[replaceIndex] = succState  
            self.rewards.T[replaceIndex] = reward   
        
        try:
            # Update the KD Tree used for nearest neighbor search
            self.kdTree = ann.kdtree(self.states.T)
        except NameError:
            pass

        # Return which state transition has been removed
        return removedState 
            
    def getStates(self):
        """ Return all states contained in this example set """
        if self.states is not None and self.states.shape[1] >= 1:
            return [State(self.states[:,i], self.stateDimensions)
                         for i in range(self.states.shape[1])]
        else:
            return []

    def getNearestNeighbor(self, state):
        """ Return the nearest neighbor of *state* in this example set """
        return list(self.getNearestNeighbors(state, 1, b=1))[0][1] # b doesn't matter

    def getNearestNeighbors(self, state, k, b):
        """ Determines *k* most similar states to the given *state*
        
        Determines *k* most similar states to the given *state*. Returns an
        iterator over (weight, neighbor), where weight is the guassian weigthed
        influence of the neighbor onto *state*. The weight is computed via
        exp(-dist/b**2)/sum_over_neighbors(exp(-dist_1/b**2)).
        Note that the weights sum to 1.
        """
        if self.states is not None:
            k = min(k, self.states.shape[1])
            
            if hasattr(self, "kdTree"): # if we can use approximate nearest neighbor
                indices, distances = self.kdTree.knn(state, k=k)
            
                # Compute weights based on distance
                weights = numpy.exp(-distances[0]/(b ** 2))
                denominator = numpy.sum(weights)
                
                # If the distances become too large, then all values can become zero
                # In this situation, we simply return the closest state and probability 1.
                if denominator == 0:
                    import warnings
                    warnings.warn("Too large distances, returning only closest example")
                    indices[0] = [indices[0][0]]
                    weights[0] = 1.0
                else:
                    # Normalize weights
                    weights = weights / denominator
                
                for index, weight in zip(indices[0], weights):
                    yield weight, State(self.states.T[index], state.dimensions)
            else:
                assert k == 1
                minDist = numpy.inf
                closestSample = None
                for index in range(self.states.shape[1]):
                    sampleState = self.states.T[index]
                    dist = numpy.linalg.norm(state - sampleState)
                    if dist < minDist:
                        minDist = dist
                        closestSample = sampleState
                yield 1.0, State(closestSample, state.dimensions)
        else:
            raise ModelNotInitialized("No state samples available")            
    
    def drawTransitions(self, samples):
        """ Returns a random iterator over the transitions
        
        Returns a random iterator over the transitions that yields *samples* 
        number of transitions from the dataset. If more samples are requested
        than contained in the data set, then data is reused.
        """
        if self.states is None:
            raise ModelNotInitialized()
        
        counter = 0
        while True:
            for i in numpy.random.permutation(range(self.states.shape[1])):
                yield (self.states[:, i],
                       self.succStates[:, i],
                       self.rewards[:, i])
                counter += 1
                if counter >= samples: return
