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
""" This module defines a multi layer perceptron (MLP) function approximator."""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from mmlf.resources.function_approximators.function_approximator import FunctionApproximator

class MLP(FunctionApproximator):
    """ Multi-Layer Perceptron function approximator. """
    
    DEFAULT_CONFIG_DICT = {}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, **kwargs):
        super(MLP, self).__init__(stateSpace)
        try:
            import ffnet
        except ImportError:
            raise Exception("Error: MLP function approximator cannot be used without the ffnet module!")
                
        self.numberOfInputs = len(stateSpace.keys())
        self.numberOfOutputs = len(actions)
        self.actions = actions
        self.stateSpace = stateSpace
        
        conec = ffnet.mlgraph((self.numberOfInputs, 8, 1))
        self.nets = dict([(action, ffnet.ffnet(conec)) for action in actions])
        for action in self.actions:
            for index in range(len(self.nets[action].weights)):
                self.nets[action].weights[index] = 0.0
    
    def computeQ(self,state, action):
        """
        Computes the Q-value of the given state, action pair
        """
        # Compute the q-value by calling the neural net for the given state,action pair.
        y = self.nets[action].call(state)[0]
        #return -math.log(max((1-y)/y, 10**-10))
        return y
    
    def train(self, trainingSet):
        """
        Train the function approximator with the given trainingSet
        """
        inputs = dict([(action,[]) for action in self.actions])
        targets = dict([(action,[]) for action in self.actions])
        for (state, action), target  in trainingSet.iteritems():
            inputs[action].append(state)
            #targets[action].append(1.0/(1 + math.exp(-(target + self.computeQ(state,action)))))
            targets[action].append(target)
                        
        for action in self.actions:
            if len(inputs[action]) > 1:
                #testInput = inputs[action][0]
                #testTarget = targets[action][0]
                #print "Before: ", -math.log((1.0-testTarget)/testTarget), self.computeQ(testInput, action)
                self.nets[action].train_bfgs(inputs[action],
                                             targets[action],
                                             bounds = [(-10,10) for weight in self.nets[action].weights],
                                             maxfun = 1000)
                #print "After: ", -math.log((1.0-testTarget)/testTarget), self.computeQ(testInput, action)
