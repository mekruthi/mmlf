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
# Created: 2009/02/02
""" Grid-based model based on a grid that spans the state space. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from math import floor, exp, sqrt, log
from collections import defaultdict
import numpy
import random

from mmlf.resources.model.model import ActionModel

# A function that computes the crossproduct of all given lists
crossProduct = lambda ss, row=[], level=0: \
                    len(ss)>1 \
                    and reduce(lambda x,y:x+y,[crossProduct(ss[1:],row+[i],level+1) for i in ss[0]]) \
                    or [row+[i] for i in ss[0]]   
       
class GridModel(ActionModel):
    """ Grid-based model based on a grid that spans the state space.
    
    The state transition probabilities and reward probabilities are 
    estimated only for the nodes of this grid. Experience samples
    are used to update the estimates of nearby grid nodes. Using a discrete
    grid has the advantage that there is only a finite amount of states for 
    which probabilities needs to be estimated.
    
    **CONFIG DICT**
        :nodesPerDim: : The number of nodes of the grid in each dimensions. The total number of grid nodes is thus nodesPerDim**dims. 
        :activationRadius: : The radius of the region around the query state in which grid nodes are activated
        :b: : Parameter controlling how fast activation decreases with distance from query state. If None, set to "activationRadius / sqrt(-log (0.01))" 
    """
    
    DEFAULT_CONFIG_DICT = {'nodesPerDim':  10,
                           'activationRadius' : 1.0,
                           'b' : None,
                           'exampleSetSize': 2500}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, nodesPerDim, activationRadius, b = None,
                 **kwargs):
        super(GridModel, self).__init__(stateSpace, **kwargs)
        
        self.dimValues = nodesPerDim
        self.activationRadius = activationRadius
        
        if b is not None:
            self.b = b
        else:
            self.b = activationRadius / sqrt(-log (0.01))
        
        # This defaultdict holds the accumulated activation
        # of the grid nodes
        self.nodeOccurences = defaultdict(float)
        # This defaultdict holds the state transition probabilities
        # of the grid
        self.successorOccurences = defaultdict(dict)
        # This defaultdict holds the expected reward of the grid nodes
        self.accumulatedReward = defaultdict(float)
        
    
    def addExperience(self, state, succState, reward):
        """
        Updates the model with one particular experience triple (s,s',r).
        (The action need not be specified since this model is 
        for one fixed action only)        
        """       
        # For all nodes of the grid that are close enough to state to
        # get activated        
        for gridNode, activation in self._getActivatedGridNodes(state):
            # Translate the successor state so that
            # (state - succState == gridNode - translatedSuccState)  
            translatedSuccState = succState + (gridNode - state)
            
            # Find the node of the grid which is the closest to 
            # translatedSuccState
            # TODO: We could do here a for loop over all grid nodes
            #       activated by translatedSuccState
            gridSuccNode = self._getClosestGridNode(translatedSuccState)
            
            #print "%s -> %s p: %s (%s -> %s)" % (gridState, gridSuccState, activation,
            #                                     state, succState) 
            
            gridNode = tuple(gridNode)
        
            # Compute the accumulated activation of this grid node
            self.nodeOccurences[gridNode] += activation
            
            # Update the state transition probability for the source state
            # gridNode
            self.successorOccurences[gridNode][gridSuccNode] = \
                    self.successorOccurences[gridNode].setdefault(gridSuccNode, 0.0) + activation
                
            # Update the reward expectation of gridNode
            self.accumulatedReward[gridNode] += reward * activation
                
    
    def getGridNodes(self):
        """ Return all nodes of this grid that have ever been activated """
        return self.nodeOccurences.iterkeys()
    
        
    def sampleState(self):
        """ Return a grid node drawn randomly """
        ## TODO: Sample according to on policy state distribution?
        return random.choice(self.nodeOccurences.keys())
    
    def sampleSuccessorState(self, state):
        """ 
        Return a grid node drawn from the state's successor distribution
        """
        succDistr = self.getSuccessorDistribution(state)
        randVal = random.uniform(0, 1)
        cumProb = 0.0
        for succNode, succProb in succDistr:
            cumProb += succProb
            if cumProb >= randVal:
                return succNode
        
    def getSuccessorDistribution(self, state):
        """ Return the successor distribution for the given state. 
        
        Returns an iterator that yields pairs of grid nodes and
        their probabilities of being the successor of the given state. 
        """
        gridNode = self._getClosestGridNode(state)
        occurences = self.nodeOccurences.get(gridNode, 0)
        
        if occurences != 0:
            for succNode, succOccurences in self.successorOccurences[gridNode].iteritems():
                succProbability = float(succOccurences)/occurences
                yield (succNode, succProbability)
    
    def getExpectedReward(self, state):
        """ Returns the expected reward for the given state """
        gridNode = self._getClosestGridNode(state)
        occurences = self.nodeOccurences[gridNode]
        if occurences == 0: # 
            return 0.0
        
        return self.accumulatedReward[gridNode] / occurences
    
    def getExplorationValue(self, state):
        """ 
        Return how often this action has been tried for the given state
        """
        gridNode = self._getClosestGridNode(state)
        return self.nodeOccurences[gridNode]        
        
    
    def _getClosestGridNode(self, state):
        """ Computes the grid node that is closest to the given state """
        gridNode = []
        for dim in range(len(state)):
            values = self.dimValues[dim]
            gridDimValue = (floor(state[dim] * values) + 0.5) / values
            gridNode.append(gridDimValue)
            
        return tuple(gridNode)
    
    def _getActivatedGridNodes(self, state):
        """ 
        Returns an iterator the yields pairs containing the        
        grid nodes activated by the given state and their activation
        """
        # Compute all grid states that are contained in the bounding box
        # around the state with edge length self.activationRadius* 2     
        gridDimRanges = []
        for dim in range(len(state)):
            minGridDimValue = int(round((state[dim] - self.activationRadius) * self.dimValues[dim])) 
            maxGridDimValue = int(round((state[dim] + self.activationRadius) * self.dimValues[dim]))
            gridDimRanges.append([float(gridDimValue - 0.5) / self.dimValues[dim] 
                                        for gridDimValue in range(minGridDimValue +1, 
                                                                  maxGridDimValue + 1)])
        
        relevantGridNodes = crossProduct(gridDimRanges)
        relevantGridNodes = map(numpy.array, relevantGridNodes)
        
        # Check for all grid nodes that are in the bounding box
        # whether they are actually contained in the spere with
        # radius self.activationRadius around state
        for gridNode in relevantGridNodes:
            distance = numpy.linalg.norm(state - gridNode)
            
            # If the grid node is close enough:
            if distance <= self.activationRadius:
                # Yield it along with its activation
                activation = exp(-(distance/self.b)**2)
                yield (gridNode, activation)
                