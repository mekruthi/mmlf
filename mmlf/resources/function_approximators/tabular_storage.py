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
""" This module defines the tabular storage function approximator

The tabular storage function approximator  can be used for discrete worlds. 
Actually it is not really a function approximator but stores the value function 
exactly.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from collections import defaultdict
from mmlf.resources.function_approximators.function_approximator import FunctionApproximator

class TabularStorage(FunctionApproximator):
    """ Function approximator for small, discrete environments.
    
    This class implements the function approximator interface. It does not really 
    approximate but simply stores the values in a table. Thus, it should not be applied 
    in environments with continuous states.
    
    **CONFIG DICT** 
        :default: : The default value that an entry stored in the function approximator has initially.
        :learning_rate: : The learning rate used internally in the updates.
    
    """
    
    DEFAULT_CONFIG_DICT = {'default':  0.0,
                           'learning_rate' : 1.0}
    
    SUPPORTED_STATE_SPACES = ["DISCRETE"]
    
    def __init__(self, actions=None, default=0, learning_rate=1.0, stateSpace=None,
                 **kwargs):
        super(TabularStorage, self).__init__(stateSpace)
        
        self.stateValueStorages = dict((action, StateValueStorage(default, learning_rate)) 
                                                    for action in actions)
        self.actions = actions
    
    def computeQ(self, state, action):
        """ Computes the Q-value of the given state, action pair """
        # Computes the Q-value by simply by looking it up in the table.If it
        # is not yet stored in the dictionary, the default value is returned.
        return self.stateValueStorages[action].values[state]
    
    def train(self, trainingSet):
        """ Train the function approximator with the given trainingSet """
        # Simply update the values in the table
        for (state, action), target in trainingSet.iteritems():
            self.stateValueStorages[action].training({state: target})
            
    def getPlainValues(self):
        """ Return a dict that maps (state, action) onto the respective value."""
        values = {}
        for action, stateValueStorage in self.stateValueStorages.iteritems():
            for state, value in stateValueStorage.values.iteritems():
                values[(state, action)] = value
        return values
    
class StateValueStorage(object):

    def __init__(self, default=0, learningRate=1.0, **kwargs):   
        # WARNING: This cannot be pickled :-(
        self.values= defaultdict(lambda : default)
        self.learningRate = learningRate
        self.default = default

    def computeQ(self, state):
        """ Compute the Q value for a given state based on this Storage. """
        return self.values[state] 
    
    def training(self, trainingSet):
        """ Train the storage with the given training set. """
        for state, target in trainingSet.iteritems():
            self.values[state] = \
                self.learningRate * target + (1 - self.learningRate) * self.computeQ(state) 
        
    def __getstate__(self):
        """ Return a pickable state for this object """
        odict = self.__dict__.copy() # copy the dict since we change it
        odict["values"] =dict(odict["values"]) # We cannot pickle defaultdict 
        return odict
    
    def __setstate__(self, dict):
        """ Return a pickable state for this object """
        # Restore attributes
        self.__dict__.update(dict)
        # Convert values-dict back to a defaultdict
        values = self.values
        self.values = defaultdict(lambda : self.default)
        self.values.update(values)
                                  
            