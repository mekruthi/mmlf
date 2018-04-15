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
"""The multilinear grid function approximator.

In this function approximator, the state space is spanned by a regular grid.
For each action a separate grid is spanned. The value of a certain state
is determined by computing the grid cell it lies in and multilinearly 
interpolating from the cell corners to the particular state. 
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import os
from math import floor
import numpy
import scipy.weave

from mmlf.resources.function_approximators.function_approximator \
            import FunctionApproximator

# A function that computes the crossproduct of all given lists
crossProduct = lambda ss, row=[], level=0: \
                    len(ss)>1 \
                    and reduce(lambda x,y:x+y,[crossProduct(ss[1:],row+[i],level+1) for i in ss[0]]) \
                    or [row+[i] for i in ss[0]]   

class MultilinearGrid(FunctionApproximator):
    """ The multilinear grid function approximator.
    
    In this function approximator, the state space is spanned by a regular grid.
    For each action a separate grid is spanned. The value of a certain state
    is determined by computing the grid cell it lies in and multilinearly 
    interpolating from the cell corners to the particular state.
    
    **CONFIG DICT** 
        :default: : The default value that an entry stored in the function approximator has initially.
        :learning_rate: : The learning rate used internally in the updates.
     
    """
    
    DEFAULT_CONFIG_DICT = {'learning_rate' : 1.0,
                           'default' : 0.0}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, learning_rate, default, **kwargs):
        super(MultilinearGrid, self).__init__(stateSpace)
        
        self.stateSpace = stateSpace
        self.actions = actions    
        self.dimValues = numpy.array([dimDescr["resolution"]
                             for dimDescr in self.stateSpace.itervalues()])
    
        self.gridValues = dict((action, dict()) for action in self.actions)
        self.learningRate = learning_rate
        self.default = default
            
    def computeQ(self, state, action):
        """
        Computes the Q-value of the given state, action pair
        """
        # Compute the grid nodes that are the corners of the cell
        # containing the given state along with their weights
        cellCorners = self._getCellCorners(state)
        
        # Compute the Q-Value of the state action pair as an weighted average
        # of the cell corners (multilinear interpolation) 
        qValue = 0.0
        for corner, weight in cellCorners:          
            qValue += self.gridValues[action].get(corner, self.default) * weight 
        
        return qValue
        
    
    def train(self, trainingSet):
        """
        Train the function approximator with the given trainingSet
        """
        # Iterate over all training samples (SxA->Q)
        for (state, action), target in trainingSet.iteritems():
#            # Compute the delta (current value - target) * learning rate
#            delta = (target - self.computeQ(state, action)) * self.learningRate
#            
#            # Compute the grid nodes that are the corners of the cell
#            # containing the given state along with their weights
#            cellCorners = self._getCellCorners(state)
#            
#            for corner, weight in cellCorners:
#                try:
#                    # TODO: Is this update rule sensible?
#                    self.gridValues[action][corner] += delta * weight
#                except:
#                    self.gridValues[action][corner] = target * weight

            # We replace the actual state by its closest grid node
            # This avoids problems with diverging Q-functions
            node = self._closestGridNode(state)
            try:
                self.gridValues[action][node] = \
                      (1 - self.learningRate) * self.gridValues[action][node] \
                                + self.learningRate * target
            except:
                self.gridValues[action][node] = target
            
    def _closestGridNode(self, state):
        """ Return the grid node with minimal distance to state """
        gridNode = []
        for dim in range(len(state)):
            values = self.dimValues[dim]
            gridDimValue = (floor(state[dim] * values) + 0.5) / values
            gridNode.append(gridDimValue)
        
        return tuple(gridNode)
    
    def _getCellCorners(self, state):
        """ Return cell corners and their weights 
        
        Return the corners of the cell containing the given state 
        and their weights (based on their relative dsitance to state
        """
        if os.sys.platform.startswith('win'):
            self._getCellCornersSlow(state)
            return
        
        dimValues = self.dimValues
        state = numpy.array(state)
        
        # Compute the indices of the corners of the cell the current state
        # is in
        dimRanges = []
        for dim in range(state.shape[0]):
            minGridDimValue = int(round(state[dim] * dimValues[dim]))
            dimRanges.append([minGridDimValue, minGridDimValue +1])
        
        # We store the indices in the variable corners and let the C-code
        # replace them in-place
        corners = numpy.array(crossProduct(dimRanges)) * 1.0
        
        # The weight of a corner is the product of the weights of this 
        # corner in each dimension. This dimension weight is
        # in principle 1 - |s[dim] - c[dim]|. Since the length of
        # an edge of a grid cell is not 1 but 1/#NumberOfCellsPerDimension,
        # we have to multiply it with this quantity.
        weights = numpy.ones(corners.shape[0]) * 1.0

        #The actual computation is done in C using the scipy.weave.inline method
        nx, ny = corners.shape
        code = \
        """
        for(int i = 0; i < nx; i++) {
          for(int j = 0; j < ny; j++) {
            corners(i, j) = float(corners(i, j) - 0.5) / dimValues(j);
            weights(i) *= (1.0 - fabs(state(j) - corners(i,j)) * dimValues(j));
          }
        }  
        """
        scipy.weave.inline(code, 
                           ["state", "corners", "dimValues", "weights", "nx", "ny"],
                           compiler='gcc', 
                           type_converters=scipy.weave.converters.blitz,
                           auto_downcast = 0)

        # We yield the corners along with their weights sequentially
        for i in range(weights.shape[0]):
            yield tuple(corners[i]), weights[i]
        
        
    def _getCellCornersSlow(self, state):
        """ Return cell corners and their weights 
        
        Return the corners of the cell containing the given state 
        and their weights (based on their relative dsitance to state
        
        Note: This is a pure-python implementation and thus slower than the 
              scipy.weave.inline version
        """
        gridDimRanges = []
        # For each dimension
        for dim in range(len(state)):
            # Compute the coordinate range in which the state falls in this dimension
            minGridDimValue = int(round(state[dim] * self.dimValues[dim]))
            gridDimRanges.append([float(gridDimValue - 0.5) / self.dimValues[dim] 
                                        for gridDimValue in range(minGridDimValue, 
                                                                  minGridDimValue + 2)])
            
        # The corners of the cells are the crossproduct of the ranges
        cellCorners = map(tuple, crossProduct(gridDimRanges))
        
        # The weight of a corner is the product of the weights of this 
        # corner in each dimension. This dimension weight is
        # in principle 1 - |s[dim] - c[dim]|. Since the length of
        # an edge of a grid cell is not 1 but 1/#NumberOfCellsPerDimension,
        # we have to multiply it with this quantity.
        prod = lambda l : reduce(lambda x,y: x*y, l)
        cellWeight = lambda corner: prod(1 - abs(state[dim] -corner[dim]) * self.dimValues[dim]
                                            for dim in range(len(state)))
        return [(corner, cellWeight(corner)) for corner in cellCorners]
