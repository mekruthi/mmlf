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
# Created: 2009/08/18
""" Function approximator based on k-Nearest-Neighbor interpolation. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from collections import defaultdict
import copy
import numpy
if "sphinx" not in __import__("sys").argv[0]: # building documentation
    import scikits.ann as ann

from mmlf.framework.state import State
from mmlf.resources.function_approximators.function_approximator \
    import FunctionApproximator

def gaussian(xSquare, b_x):
    return numpy.exp(-xSquare/(b_x ** 2))

class KNNFunctionApproximator(FunctionApproximator):
    """ Function approximator based on k-Nearest-Neighbor interpolation

    A function approximator that stores the a given set of 
    (state, action) -> Q-Value samples. The sample set is split into subsets, one  
    for each action (thus, a discrete, finite set of actions is assumed). When the  
    Q-Value of a state-action is queried, the *k* states most similar to the query 
    state are extracted (under the constraint that the query action is the action  
    applied in these states). The Q-Value of the query state-action is computed as  
    weighted linear combination of the *k* extracted samples, where the weighting is 
    based on the distance between the respective state and the query state. The  
    weight of a sample is computed as exp(-(distance/b_x)**2), where *b_x* is an 
    parameter that influences the generalization breadth. Smaller values of *b_X* 
    correspond to increased weight of more similar states.
    
    **CONFIG DICT** 
        :k: : The number of neighbors considered in k-Nearest Neighbors
        :b_X: : The width of the gaussian weighting function. Smaller values of b_X correspond to increased weight of more similar states. 
    """
    
    DEFAULT_CONFIG_DICT = {'k':  10,
                           'b_X' : 0.1}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, k = 10, b_X = 0.1, **kwargs):        
        super(KNNFunctionApproximator, self).__init__(stateSpace)
        
        self.stateSpace = stateSpace
        self.actions = actions
        self.k = k
        self.b_X = b_X
        
        
        self.actionsKDTree = {}
        self.states = {}
    
    def computeQ(self, state, action):
        """ Computes the Q-value of the given state-action pair
        
        The Q-Value of the query state-action is computed as  weighted linear 
        combination of the *k* nearest neighbors, where the weighting is based
        on the distance between the respective state and the query state. 
        """
        if not action in self.actionsKDTree \
            or self.actionsKDTree[action] == None:
            return 0.0
       
        k = min(self.k, self.states[action].shape[0])
        indices, distances = self.actionsKDTree[action].knn(state, k)
        
        qValue = 0.0
        denominator = 0.0
        for index, distance in zip(indices[0], distances[0]):
            neighbor = State(self.states[action][index],
                             state.dimensions)
            neighborsQValue = self.qValues[(neighbor, action)]
            
            weight = gaussian(distance, self.b_X)
            qValue += weight * neighborsQValue
            denominator += weight
        
        return qValue / denominator
    
    def train(self, trainingSet):
        """ Trains the KNN function approximator with the given *trainingSet*. 
        
        Stores the examples contained in the *trainingSet* and overwrites
        the old examples
        """
        self.qValues = trainingSet 
        
        states = defaultdict(list)
        for (state, action), target in trainingSet.iteritems():
            states[action].append(state)

        for action in self.actions:
            if len(states[action]) > 0:
                self.states[action] = numpy.vstack(states[action])                    
                self.actionsKDTree[action] = ann.kdtree(self.states[action])
            else:
                self.actionsKDTree[action] = None
                
    def __getstate__(self):
        # ann.kdTree instances  cannot be pickled, so we remove them from the 
        # object before pickling it
        pickleDict = copy.copy(self.__dict__)
        for action in self.actions:
            pickleDict["actionsKDTree"][action] = None
        return pickleDict
    
    def __setstate__(self, dict):
        self.__dict__ = dict
        # restore 
        for action in self.actions:
            self.actionsKDTree[action] = ann.kdtree(self.states[action])
            