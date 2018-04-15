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
# Created: 2009/07/08
""" Policies for discrete and continuous action spaces based on an MLP 

This module contains classes that represent policies for discrete and
continuous action spaces that are based on an multi-layer perceptron
representation.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random

from mmlf.resources.policies.policy import Policy

class MLPPolicy(Policy):
    """ Policy based on a MLP representation for disc. and cont. action spaces
    
    The MLP are based on the ffnet module. It can be
    specified how many *hiddenUnits* should be contained in the MLP and
    whether the neurons should get an addition *bias* input. If 
    *independentOutputs* is set to True, for each output, the hidden layer is 
    cloned, i.e. different outputs do not share common neurons in the network.    
    The *actionSpace* defines which actions the agent can choose.
     
    If the action space has no continuous actions, the finite set of (n) 
    possible action selections is determined. Action
    selection is based on a 1-of-n encoding, meaning that for each
    available action the MLP has one output. The action whose corresponding
    network output has maximal activation is chosen.
    
    If the action space has continuous actions, it is assumed that the action 
    space is one-dimensional. The MLP hase one output falling 
    in the range [0,1]. This output is scaled to the allowed action range to
    yield the action. Currently, it is assumed that continuous action spaces are 
    one-dimensional and contiguous. 
    
    **CONFIG DICT**
        :bias: : Determines whether or not an additional bias (state dimension always equals to 1) is added 
        :hiddenUnits: : Determines hte number of neurons in the hidden layer of the multi-layer perceptron
        :independentOutputs: : If True, for each output the hidden layer is cloned, i.e. different outputs do not share common neurons in the network        
    """
    
    DEFAULT_CONFIG_DICT = {'hiddenUnits':  5,
                           'bias' : True,
                           'independentOutputs': False}
    
    def __init__(self, inputDims, actionSpace, hiddenUnits=5, bias=True,
                 independentOutputs=False, **kwargs):
        assert not actionSpace.hasContinuousDimensions() \
                or actionSpace.getNumberOfDimensions() == 1, \
                "MLP policy can currently not deal with continuous action "\
                "spaces with more than one dimension!" 
                
        try:
            import ffnet
        except:
            import warnings
            warnings.warn("The MLP policy module requires the ffnet package.")
        
        self.continuousActions = actionSpace.hasContinuousDimensions()
        
        if self.continuousActions:
            actionDimension = actionSpace.getDimensions()[0] # there is per assert only 1
            actionRanges = actionDimension.getValueRanges()
            assert len(actionRanges) == 1, "MLP policy cannot deal with "\
                                           "non-contiguous action ranges."
            self.actionRange = actionRanges[0]
            self.numActions = 1 # TODO
        else:
            self.actions = actionSpace.getActionList()
            self.numActions = len(self.actions)
        
        self.inputDims = inputDims
        self.hiddenUnits = hiddenUnits
        self.bias = bias
        
        # Determine network topology
        if independentOutputs:
            conec = ffnet.imlgraph((inputDims, hiddenUnits, self.numActions),
                                   biases=self.bias)
        else:
            conec = ffnet.mlgraph((inputDims, hiddenUnits, self.numActions),
                                   biases=self.bias)
        
        # Create net based on connectivity
        self.net = ffnet.ffnet(conec)
        
    def evaluate(self, state):
        """ Evaluates the policy for the given state """
        # Compute the output of the network for the given state
        actionActivation = self.net(state)
        
        if not self.continuousActions:
            # Greedy action selection
            selectedAction = max(zip(actionActivation, 
                                     range(len(actionActivation))))[1]
                                     
            return self.actions[selectedAction]
        else:
            # Scale action activation from [0, 1] to self.actionRange
            action = actionActivation[0]*(self.actionRange[1] - self.actionRange[0]) \
                         + self.actionRange[0]
            return [action]
    
    def getParameters(self):
        """ Returns the parameters of this policy """
        return self.net.weights
    
    def setParameters(self, parameters):
        """ Sets the parameters of the policy to the given parameters """
        self.net.weights = parameters
        
    def getParameterSampleFunction(self):
        """Returns a function that samples weight vectors for this policy class """
        return lambda : [random.random() 
                            for i in range(len(self.net.weights))]
    
         
        