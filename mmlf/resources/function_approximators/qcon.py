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

# Author: Larbi Abdenebaoui
# Created: 2008
"""This module defines the QCON function approximator """

__author__ = "Larbi Abdenebaoui"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Jan Hendrik Metzen', 'Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from mmlf.resources.function_approximators.function_approximator import FunctionApproximator

class QCON(FunctionApproximator):
    """ Function approximator based on the connectionist QCON architecture
    
    
    This class implements the QCON architecture which consists of a connectionist Q-learning model
    where each action has a separate network.
    The feed-forward neural network are implemented using the python package ffnet
    
    **CONFIG DICT**
        :hidden: : The number of neurons in the hidden layer of the multi-layer perceptron
        :learning_rate: : The learning rate used internally in the updates.
    """
    
    DEFAULT_CONFIG_DICT = {'hidden':  3,
                           'learning_rate' : 1.0}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, hidden, 
                 learning_rate, **kwargs):
        super(QCON, self).__init__(stateSpace)
        try:
            import ffnet
        except ImportError:
            raise Exception("Error: MLP function approximator cannot be used without the ffnet module!")
    
        self.learning_rate = learning_rate 
        self.hidden=hidden    
            
        self.numberOfInputs = len(stateSpace.keys())
        self.numberOfOutputs = len(actions)
        self.actions = actions
       
        
        conec = ffnet.mlgraph((self.numberOfInputs, self.hidden, 1))
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
            targets[action].append(target)
            
        for action in self.actions:
            if len(inputs[action]) > 1:
                testInput = inputs[action][0]
                testTarget = targets[action][0]
                self.nets[action].train_momentum(inputs[action],
                                                 targets[action],
                                                 self.learning_rate, 
                                                 0.80000000000000004,# momentum 
                                                 100,# the maximum nbr of iteraions
                                                 0 #disp print convergence message if 
                                                 #non-zero (default is 0) 
                                                 )
   
