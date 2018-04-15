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
# Created: 2009/06/30
""" Linear Policies for discrete and continuous action spaces

This module contains classes that represent linear policies for discrete and
continuous action spaces.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import numpy

from mmlf.resources.policies.policy import Policy 
from mmlf.framework.state import State
from mmlf.framework.spaces import Dimension

class LinearDiscreteActionPolicy(Policy):
    """ Linear policy for discrete action spaces
    
    Class for linear policies on discrete action space using a 1-of-n encoding,
    i.e. \pi(s) = argmax_{a_j} \sum_{i=0}^n w_ij s_i
    
    For each discrete action, *numOfDuplications* outputs in the
    1-of-n encoding are created, i.e. n=numOfActions*numOfDuplications.
    This allows to represent more complex policies.
    
    Expected parameters:
     * *inputDims*: The number of input (state) dimensions
     * *actionSpace*: The action space which determines which actions are allowed.
     
    **CONFIG DICT**
        :bias: : Determines whether or not an additional bias (state dimension always equals to 1) is added 
        :numOfDuplications: : Determines how many outputs there are for each discrete action  in the 1-of-n encoding.                            
    """
    
    DEFAULT_CONFIG_DICT = {'bias':  True,
                           'numOfDuplications' : 1}
    
    def __init__(self, inputDims, actionSpace, bias = True,
                 numOfDuplications=1, **kwargs):
        assert not actionSpace.hasContinuousDimensions() 
        self.inputDims = inputDims
        self.actions = actionSpace.getActionList() * numOfDuplications
        self.numActions = len(self.actions)
        self.bias = bias
        if self.bias:
            self.inputDims += 1
            
        # Sample some initial weights
        self.weights = self.getParameterSampleFunction()() 

    def evaluate(self, state):
        """ Evaluates the policy for the given state """
        # If bias is desired, we simply append an additional dimension that 
        # always takes the value 1
        if self.bias:
            dimensions = [dimension for dimension in state.dimensions] 
            biasDimension = Dimension("zzz_bias", "continuous", [[0,1]])
            dimensions.append(biasDimension)
            input = State(numpy.hstack((state, [1])), dimensions)
        else: # Just create a copy of the state
            input = State(state, state.dimensions)
        
        # Scale state dimensions to range  (-1, 1)
        input.scale(-1,1)
        
        # Compute the activation (the preference of the policy) for each action
        # The last action has always activation 0 (remove redundant 
        # representations for the same policy)
        actionActivations = []
        for actionIndex in range(self.numActions - 1):
            activation = numpy.dot(self.weights[self.inputDims * actionIndex:
                                                self.inputDims * (actionIndex + 1)],
                                    input)
            actionActivations.append(activation)            
        actionActivations.append(0.0)
        
        # Greedy action selection
        selectedAction = max(zip(actionActivations, 
                                 range(len(actionActivations))))[1]
        
        return self.actions[selectedAction]
    
    def getParameters(self):
        """ Returns the parameters of this policy """
        return self.weights
    
    def setParameters(self, parameters):
        """ Sets the parameters of the policy to the given parameters """
        self.weights = parameters
    
    def getParameterSampleFunction(self):
        """Returns a function that samples weight vectors for this policy class """
        # The last action is assumed to have always zero activation so we do not 
        # create any weights for it
        return lambda : numpy.random.random(size = self.inputDims * (self.numActions - 1)) 

class LinearContinuousActionPolicy(Policy):
    """ Linear policy for continuous action spaces
    
    Class for linear policies on continuous action space ,
    i.e. \pi(s) = [\sum_{i=0}^n w_i0 s_i, \dots, \sum_{i=0}^n w_ij s_i]
    
    Expected parameters:
     * *inputDims*: The number of input (state) dimensions
     * *actionSpace*: The action space which determines which actions are allowed.
                      It is currently only possible to use this policy with
                      one-dimensional action space with contiguous value ranges

    **CONFIG DICT**
        :bias: : Determines whether or not an additional bias (state dimension 
                 always equals to 1) is added  
    """
    
    DEFAULT_CONFIG_DICT = {'bias':  True}
    
    def __init__(self, inputDims, actionSpace, bias=True, **kwargs):
        assert actionSpace.hasContinuousDimensions() 
    
        self.bias = bias
        self.inputDims = inputDims
        if self.bias:
            self.inputDims += 1
        
        for actionDimension in actionSpace.getDimensions(): 
            actionRanges = actionDimension.getValueRanges()
            assert len(actionRanges) == 1, "Linear policy cannot deal with "\
                                           "non-contiguous action ranges."
        self.numActions = len(actionSpace.getDimensions())

        self.weights = self.getParameterSampleFunction()() 
        
    def evaluate(self, state):
        """ Evaluates the policy for the given state """
        # If bias is desired, we simply append an additional dimension that 
        # always takes the value 1
        if self.bias:
            dimensions = [dimension for dimension in state.dimensions] 
            biasDimension = Dimension("zzz_bias", "continuous", [[0,1]])
            dimensions.append(biasDimension)
            state = State(numpy.hstack((state, [1])), dimensions)
            
        # Scale state dimensions to range  (-1, 1)
        state.scale(-1,1)
        
        # Compute the activation (the preference of the policy) for each action
        output = []
        for outputDimIndex in range(self.numActions):
            activation = numpy.dot(self.weights[self.inputDims * outputDimIndex:
                                                self.inputDims * (outputDimIndex + 1)],
                                    state)
            output.append(activation)
            
        return output
    
    def getParameters(self):
        """ Returns the parameters of this policy """
        return self.weights
    
    def setParameters(self, parameters):
        """ Sets the parameters of the policy to the given parameters """
        self.weights = parameters
        
    def getParameterSampleFunction(self):
        """Returns a function that samples weight vectors for this policy class """
        return lambda : numpy.random.random(size = self.inputDims * self.numActions) 
        