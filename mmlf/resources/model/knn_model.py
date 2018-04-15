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
# Created: 2009/08/03
""" An action model class based on KNN state transition modeling.
    
This model is based on the model proposed in:
Nicholas K. Jong and Peter Stone,
"Model-based function approximation in reinforcement learning",
in Proceedings of the 6th international joint conference on Autonomous agents and multiagent systems 
Honolulu, Hawaii: ACM, 2007, 1-8, http://portal.acm.org/citation.cfm?id=1329125.1329242.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
import numpy

if "sphinx" not in __import__("sys").argv[0]: # building documentation
    import scikits.ann as ann

from mmlf.framework.state import State
from mmlf.resources.model.model import ActionModel, ModelNotInitialized

def gaussian(xSquare, b):
    return numpy.exp(-xSquare/(b ** 2))

class KNNModel(ActionModel):
    """ An action model class based on KNN state transition modeling.
    
    This model learns the state successor (and predecessor)
    function using the "k-Nearest Neighbors" (KNN) regression 
    learner. This learner learns a stochastic model, mapping each state $s$ to the
    successor $s' = s + (s^'_{neighbor} - s_{neighbor})$ with probability
    exp(-(||s - s_{neighbor}||/b_Sa)^2)/ sum_{neighbor \in knn(s)} exp(-(||s - s_{neighbor}||/b_Sa)^2).
    The reward function is learned using an KNN model, too.
    
    **CONFIG DICT**
        :exampleSetSize: : The maximum number of example transitions that is remembered in the example set. If the example set is full, old examples must be deleted or now new examples are accepted.
        :k: : The number of neighbors considered in k-Nearest Neighbors
        :b_Sa: : The width of the gaussian weighting function. Smaller values of *b_Sa* correspond to increased weight of more similar states
    """
    
    DEFAULT_CONFIG_DICT = {'k':  10,
                           'b_Sa' : 0.1,
                           'exampleSetSize': 2500}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, k=10, b_Sa=0.1, **kwargs):
        super(KNNModel, self).__init__(stateSpace, **kwargs)
        
        self.k = k
        self.b_Sa = b_Sa
        
        self.states = None
        self.succStates = None
        
        self.successorSamples = dict()
        self.predecessorSamples = dict()
        
        self.succKDTree = None
        self.rebuildSucc = True
        
        self.predKDTree = None
        self.rebuildPred = True
    
    def addExperience(self, state, succState, reward):
        """ Add the given experience tuple to the example state transitions-
        
        Updates the model with one particular experience triple (s,s',r).
        (The action need not be specified since this model is 
        for one fixed action only).
        """
        if self.states == None:
            self.states = numpy.array(numpy.atleast_2d(state))
            self.succStates = numpy.array(numpy.atleast_2d(succState))
        else:
            self.states = numpy.vstack((self.states, state))
            self.succStates = numpy.vstack((self.succStates, succState))

        self.successorSamples[state] = (succState, reward)
        self.predecessorSamples[succState] = (state, reward)
        
        self.rebuildSucc = True
        self.rebuildPred = True
    
    def sampleState(self):
        """ Return a known state randomly sampled with uniform distribution"""
        return random.choice(self.successorSamples.keys())
    
    def sampleSuccessorState(self, state):
        """ Return a states drawn from *state*'s successor distribution 
        
        Returns a possible successor state of *state* drawn from the successor 
        state distribution according to its probability mass function.
        """
        if self.states == None:
            raise ModelNotInitialized()
        
        succDistr = self.getSuccessorDistribution(state)
        randVal = random.uniform(0, 1)
        cumProb = 0.0
        for succState, succProb in succDistr:
            cumProb += succProb
            if cumProb >= randVal:
                return succState
            
        assert False, "No successor state has been found!"
        
    def getSuccessorDistribution(self, state):
        """ Return the successor distribution for the given *state*. 
        
        Returns an iterator that yields pairs of states and
        their probabilities of being the successor of the given *state*. 
        """
        if self.states == None:
            raise ModelNotInitialized()
        
        k = min(self.states.shape[0], self.k)
        
        if self.rebuildSucc:
            self.succKDTree = ann.kdtree(self.states)
            self.rebuildSucc = False
            
        indices, distances = self.succKDTree.knn(state, k)
        
        denominator = numpy.sum(numpy.exp(-distances[0]/(self.b_Sa ** 2)))
        
        # If the distances become too large, then all values can become zero
        # In this situation, we simply return the closest state and probability 1.
        if denominator == 0 or numpy.isnan(denominator):
            import warnings
            warnings.warn("Too large distances, returning only closest example")
            indices[0] = [indices[0][0]]
            distances[0] = [0.0] 
            denominator = numpy.exp(0.0/(self.b_Sa ** 2))

        for index, distance in zip(indices[0], distances[0]):
            neighbor = State(self.states[index], state.dimensions) # TODO: not use state.dimensions
            succState, reward = self.successorSamples[neighbor]
            
            delta = succState - neighbor
            predictedSuccState = State(state + delta, state.dimensions)
            
            if not 0 <= gaussian(distance, self.b_Sa) / denominator <= 1:
                import warnings
                import sys
                warnings.warn("Invalid distances in KNN Model!")
                print distances
                sys.exit(0)
            
            yield predictedSuccState, gaussian(distance, self.b_Sa) / denominator

    def samplePredecessorState(self, state):
        """ Return a states drawn from *state*'s predecessor distribution 
        
        Returns a possible predecessor state of *state* drawn from the 
        predecessor state distribution according to its probability mass function.
        """
        if self.succStates == None:
            raise ModelNotInitialized()
        
        predDistr = self.getPredecessorDistribution(state)
        randVal = random.uniform(0, 1)
        cumProb = 0.0
        for predState, predProb in predDistr:
            cumProb += predProb
            if cumProb >= randVal:
                return predState
        
        assert False, "No predecessor state has been found!"
        
    def getPredecessorDistribution(self, state):
        """ Return a states drawn from *state*'s predecessor distribution 
        
        Returns a possible predecessor state of *state* drawn from the 
        predecessor state distribution according to its probability mass function.
        """
        if self.succStates == None:
            raise ModelNotInitialized()
        
        k = min(self.states.shape[0], self.k) 
        
        if self.rebuildPred:
            self.predKDTree = ann.kdtree(self.succStates)
            self.rebuildPred = False        

        indices, distances = self.predKDTree.knn(state, k)
        
        denominator = numpy.sum(numpy.exp(-distances[0]/(self.b_Sa ** 2)))
        
        # If the distances become too large, then all values can become zero
        # In this situation, we simply return the closest state and probability 1.
        if denominator == 0:
            import warnings
            warnings.warn("Too large distances, returing only closest example")
            indices[0] = [indices[0][0]]
            distances[0] = [0.0] 
            denominator = numpy.exp(0.0/(self.b_Sa ** 2))

        for index, distance in zip(indices[0], distances[0]):
            neighbor = State(self.succStates[index], state.dimensions) # TODO: not use state.dimensions 
            predState, reward = self.predecessorSamples[neighbor]
            
            delta = predState - neighbor
            predictedPredState = State(state + delta, state.dimensions)
            
            yield predictedPredState, gaussian(distance, self.b_Sa) / denominator
    
    def getExpectedReward(self, state):
        """ Returns the expected reward for the given state """
        if self.states == None:
            return 0.0
        
        k = min(self.states.shape[0], self.k) 
        
        if self.rebuildSucc:
            self.succKDTree = ann.kdtree(self.states)
            self.rebuildSucc = False

        indices, distances = self.succKDTree.knn(state, k)
        
        denominator = numpy.sum(numpy.exp(-distances[0]/(self.b_Sa ** 2)))
        
        # If the distances become too large, then all values can become zero
        # In this situation, we simply return the closest state and probability 1.
        if denominator == 0:
            import warnings
            warnings.warn("Too large distances, returning only closest example")
            indices[0] = [indices[0][0]]
            distances[0] = [0.0]
            denominator = numpy.exp(0.0/(self.b_Sa ** 2))

        expectedReward = 0.0
        for index, distance in zip(indices[0], distances[0]):
            neighbor = State(self.states[index], state.dimensions) # TODO: not use state.dimensions
            
            succState, reward = self.successorSamples[neighbor]
            
            weight = gaussian(distance, self.b_Sa) / denominator
            expectedReward += reward * weight
            
        return expectedReward
    
    def getExplorationValue(self, state):
        """ Return the exploratory value of the given state *state*
        
        The exploratory value of a state under this model is defined simply as
        the sum of the activations of its k nearest neighbors 
        """
        if self.states == None:
            return 0.0
        
        k = min(self.states.shape[0], self.k)            
        
        if self.rebuildSucc:
            self.succKDTree = ann.kdtree(self.states)
            self.rebuildSucc = False
            
        indices, distances = self.succKDTree.knn(state, k)
        
        return numpy.sum(numpy.exp(-distances[0]/(self.b_Sa ** 2)))  
