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

""" Module that contains the main State class of the MMLF
    
The MMLF uses a state class that is derived from numpy.array. In contrast
to numpy.array, MMLF states can be hashed and thus be used as keys for 
dictionaries and set. This is realized by calling 
"hashlib.sha1(self).hexdigest()"
In order to improve performance, State classes cache their hash value. Because
of this, it is necessary to consider a state object to be constant, i.e. not
changeable (except for calling "scale").

Each state object stores its state dimension definition. Furthermore, it 
implements a method "scale(self, minValue = 0, maxValue = 1):" which scales each
dimension into the range (minValue, maxValue).
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import copy
import hashlib
import numpy

class State(numpy.ndarray):
    """ State class for the MMLF.
    
    Based on numpy arrays, but can be used as dictionary key
    """
    
    def __new__(subtype, inputArray, dimensions):
        # Convert inputArray to array if not already the case
        if type(inputArray) != numpy.ndarray:
            inputArray = numpy.array(inputArray)
        
        assert inputArray.shape[0] == len(dimensions), \
               "Number of values and dimensions is not equal!"
        # Refer to http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        # for infos about subclassing ndarray
         
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(inputArray, dtype=numpy.float64).view(subtype)
        
        # Add new attributes to object
        obj.dimensions = dimensions
        
        obj.dimensionNames = [dimension["dimensionName"] for dimension in obj.dimensions]
        assert sorted(obj.dimensionNames) == obj.dimensionNames, \
                 "Dimensions must be sorted alphabetically according to their names!"
        # A mapping from query dimension to index of this dimension
        obj.nameToIndex = dict((dimensionName, index) 
                                    for index, dimensionName in enumerate(obj.dimensionNames))
        # If index is already queried, this one is returned
        for index in obj.nameToIndex.values():
            obj.nameToIndex[index] = index
        
        obj.hashCache = None   
        
        # Finally, we must return the newly created object:
        return obj
    
    def __reduce__(self):
        """
        Refer to
        http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
        for infos about pickling ndarray subclasses
        """
        object_state = list(super(State, self).__reduce__())        
        object_state[2] = (object_state[2], self.dimensions, self.dimensionNames,
                           self.nameToIndex)
        return tuple(object_state)
    
    def __setstate__(self, state):
        nd_state, dimensions, dimensionNames, nameToIndex = state
        super(State, self).__setstate__(nd_state)
        
        self.dimensions = dimensions
        self.dimensionNames = dimensionNames
        self.nameToIndex = nameToIndex
    
    # Methods for hashing State objects 
    def __hash__(self):
        # State arrays should be potential dictionary keys
        # Cache hash values (performance critical)
        if not hasattr(self, "hashCache") or self.hashCache == None :
            self.hashCache = int(hashlib.sha1(self).hexdigest(), 16)
        
        return self.hashCache
        
    # State arrays should be potential dictionary keys and implement all 
    # comparison operators
    def __eq__(a, b):
        return hash(a) == hash(b)
    
    def __ne__(a, b):
        return hash(a) != hash(b)
       
    def __cmp__(a, b):
        return cmp(hash(a), hash(b)) 
     
    def __lt__(a, b):
        return hash(a) < hash(b)
     
    def __gt__(a, b):
        return hash(a) > hash(b)
    
    def __le__(a, b):
        return hash(a) <= hash(b)
     
    def __ge__(a, b):
        return hash(a) >= hash(b)
     
    def __getitem__(self, key):
        try:
            return super(State, self).__getitem__(self.nameToIndex[key])
        except KeyError: # If accessing the array with a key like -1
            return super(State, self).__getitem__(key)
            
    def __setitem__(self, key, value):
        # VERY important: clean hash cache since object might change
        self.hashCache = None
        try:
            super(State, self).__setitem__(self.nameToIndex[key], value)
        except KeyError: # If accessing the array with a key like -1
            super(State, self).__setitem__(key, value)
        
    def hasDimension(self, dimension):
        """ Returns whether this state has the dimension *dimension*"""
        return dimension in self.dimensions
    
    def scale(self, minValue = 0, maxValue = 1):
        """ Scale state such that it falls into the range (minValue, maxValue)
        
        Scale the state so that (for each dimension) the range specified
        in this state space is scaled to the interval (minValue, maxValue)
        """
        # Create a copy to avoid side-effects
        self.dimensions = copy.deepcopy(self.dimensions)
        # Clear cached hash value
        self.hashCache = None   
        
        for dimension, index in zip(self.dimensions, range(self.shape[0])):
            #If the state dimension is discrete: do no scaling at all
            if dimension.isDiscrete(): 
                pass
            else: # Else: Scale
                rangeMin, rangeMax = dimension.getValueRanges()[0]
                # Scale to [0,1]
                self[index] = (self[index] - rangeMin) / (rangeMax - rangeMin)
                # Scale to [minValue, maxValue]
                self[index] = self[index] * (maxValue - minValue) + minValue
                
                # Update dimension definition
                dimension["dimensionValues"] = [(minValue, maxValue)]
    
    def isContinuous(self):
        """ Return whether this state is in a continuous state space """
        for dimension in self.dimensions:
            if dimension.isContinuous():
                return True
        return False
                