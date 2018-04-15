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
"""The Cerebellar Model Articulation Controller (CMAC) function approximator. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import copy
import random
import numpy
from collections import deque

from mmlf.resources.function_approximators.function_approximator \
                    import FunctionApproximator
from mmlf.framework.state import State 
            
class CMAC(FunctionApproximator):
    """ The Cerebellar Model Articulation Controller (CMAC) function approximator.
    
    **CONFIG DICT** 
        :number_of_tilings: : The number of independent tilings that are used in each tile coding
        :default: : The default value that an entry stored in the function approximator has initially
        :learning_rate: : The learning rate used internally in the updates
    
    """
    
    DEFAULT_CONFIG_DICT = {'number_of_tilings':  10,
                           'learning_rate' : 1.0,
                           'default' : 0.0}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, number_of_tilings, 
                 learning_rate, default, **kwargs):
        super(CMAC, self).__init__(stateSpace)
        
        self.stateSpace = stateSpace
        self.actions = actions
        
        self.numberOfTilings = number_of_tilings
        
        self.default = float(default)
        
        #Create a Tile Coding for each possible action
        self.tileCoding = dict()
        
        #For all possible actions:
        for action in actions:
            default = self.default / self.numberOfTilings
            self.tileCoding[action] = TileCoding(stateSpace, 
                                                 self.numberOfTilings,
                                                 learning_rate, default)
            
    def computeQ(self, state, action):
        """ Computes the Q-value of the given state, action pair
        
        It is assumed that a state is a n-dimensional vector 
        where n is the dimensionality of the state space. Furthmore, the
        states must have been scaled externally so that the value of each dimension
        falls into the bin [0,1]. action must be one of the actions given to the constructor
        """
        # Delegate to the responsible tile coding
        return self.tileCoding[action].computeQ(state)
    
    def train(self, trainingSet):
        """ Trains the function approximator using the given training set.
        
        trainingSet is a dictionary containing training data where the 
        key is the respective (state, action) pair whose Q-value should
        be updated and the dict-value is this desired Q-value.
        """
        #Split the training set according to the action they correspond to
        trainingSets = dict([(action, dict()) for action in self.actions])
        for (state, action), target in trainingSet.iteritems():
            trainingSets[action][state] = target
            
        for action in self.actions:
            self.tileCoding[action].training(trainingSets[action])
                

class TileCoding(object):
    """ A class which implements a Tile Coding (a set of superimposed tilings).
    
    A TileCoding with numberOfTilings overlaid tilings. Each dimension is 
    assumed to be continuous with values in the range 0,1. The tiles have equal 
    width and the tilings are overlaid with equidistant offset.
    """    
    def __init__(self, stateSpace, numberOfTilings, learningRate, default):
        self.numberOfTilings = numberOfTilings
        self.learningRate = learningRate
        self.default = default
        
        self.tilings = [Tiling(stateSpace, index) 
                                    for index in range(numberOfTilings)]
                      
               
    def computeQ(self, state):
        """ Compute the q value for a given state based on this TileCoding.
        
        Q-Values are computed by simply summing up the weights of the tiles,
        which are "activated" by the given state
        """
        q_value = 0.0
        for tiling in self.tilings:
            #Add the weight of the activated tile to the Q value.
            q_value += tiling.get(state, self.default) #If never set: return default

        return q_value
    
    def training(self, trainingSet):
        """
        Train the TileCoding with the given training set using the 
        exaggerator update rule w(i) = w(i) + alpha * (y - Q(x))
        """
        for state, target in trainingSet.iteritems():
            currentQ = self.computeQ(state)
            deltaTiling = (target - currentQ) / self.numberOfTilings
            for tiling in self.tilings:
                tile = tiling.getTile(state)
                # We have to call dicts methods since we overwrote get and __setitem__
                # for translating states to tiles automatically
                tileTarget = super(Tiling, tiling).get(tile, self.default) + deltaTiling
           
                weight = super(Tiling, tiling).get(tile, tileTarget)
                
                # Compute the new value of the tile
                value = (1 - self.learningRate) * weight + self.learningRate * tileTarget
                
                #Update its Q value
                super(Tiling, tiling).__setitem__(tile, value)


class Tiling(dict):
    """ Single tiling (regular grid in a space with specified number of dims). """
   
    def __init__(self, stateSpace, index):
        self.index = index
        
        self.tilesPerDimension = numpy.zeros(stateSpace.getNumberOfDimensions())
        self.dimensionNames = []
        index = 0
        for dimName, dimDescr in sorted(stateSpace.items()):
            self.dimensionNames.append(dimName)
            self.tilesPerDimension[index] = dimDescr["resolution"]
            index += 1
            
        self.offset = numpy.array([random.uniform(0.0, 1.0/tiles) 
                                    for tiles in self.tilesPerDimension])
        
        # A dictionary mapping states to tiles (improves speed)
        # To save memory, only the tiles for the most recent states are memoized.
        self.stateToTileCache = dict()
        self.recentStatesOrder = deque()
        
    def __setitem__(self, key, value):
        """ Set the entry of this key to the given value """
        #Compute the activated tile in this dimension
        tile = self.getTile(key)
        
        #Call dicts set method
        return super(Tiling,self).__setitem__(tile, value)
    
    def get(self, key, default = 0):
        """
        Get the value for this key, if entry for this key has been added to
        the tiling. Otherwise return default.
        """
        #Compute the activated tile in this dimension
        tile = self.getTile(key)
        
        #Call dicts get method
        return super(Tiling,self).get(tile, default) #If not found: return default
    
    def getTile(self, state):
        """ Compute the activated tile for the given state_value """
        if state in self.stateToTileCache:
            return self.stateToTileCache[state]
        else:
            scaledState = State(state, copy.copy(state.dimensions)) # avoid side-effects
            scaledState.scale(0, 1)
            tile = tuple(numpy.round((numpy.array(scaledState) + self.offset)
                                     *  self.tilesPerDimension).astype(numpy.int))
            self.stateToTileCache[state] = tile
            self.recentStatesOrder.appendleft(state)
            if len(self.recentStatesOrder) > 50:
                oldestState = self.recentStatesOrder.pop()
                self.stateToTileCache.pop(oldestState)
            return tile
    
