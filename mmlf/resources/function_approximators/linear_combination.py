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
# Created: 2009/03/01
"""The linear combination function approximator

This module defines the linear combination function approximator.
It computes the Q-value as the dot product of the feature vector
and a weight vector. It's main application area are discrete worlds;
however given appropriate features it can also be used in continuous
world.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from mmlf.resources.function_approximators.function_approximator import FunctionApproximator
    
class LinearCombination(FunctionApproximator):
    """ The linear combination function approximator.
     
    This class implements the function approximator interface.
    It computes the Q-value as the dot product of the feature vector
    and a weight vector. It's main application area are discrete worlds;
    however given appropriate features it can also be used in continuous
    world. At the moment, it ignores the planned action since it is assummed
    that it is used in combination with minmax tree search.
    
    **CONFIG DICT** 
        :learning_rate: : The learning rate used internally in the updates.
    """
    
    DEFAULT_CONFIG_DICT = {'learning_rate' : 1.0}
    
    SUPPORTED_STATE_SPACES = ["DISCRETE", "CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, learning_rate = 1.0, **kwargs):
        super(LinearCombination, self).__init__(stateSpace)
        
        self.stateSpace = stateSpace
        self.actions = actions    
    
        self.featureWeights = None
        self.learningRate = learning_rate
    
    def computeQ(self, features, action):
        """ Computes the Q-value of the given state, action pair
        
        The Q-value is computed as the dot product of the feature vector
        and a weight vector. At the moment, the planned action is ignored 
        since it is assummed that it is used in combination with
        minmax tree search.        
        """
        # Lazily create the weight vector as soon as the dimensionality  
        # is known
        if self.featureWeights == None:
           self.featureWeights = [0.0 for i in range(len(features))] 
        # Ignore the action and compute the weights as a linear 
        # combination of the features
        value = sum(feature * weight for feature, weight in zip(features, 
                                                                self.featureWeights))
        return value 
    
    def train(self, trainingSet):
        """ Train the function approximator with the given trainingSet """
        # Simply update the values in the table
        for (feature, action), value in trainingSet.iteritems():
            error = self.computeQ(feature, action) - value
            #print "self.computeQ(state, action) %s  value %s" % (self.computeQ(state, action),  value) 
            featureSum = sum(feature)
            #print "stateSum %s"% stateSum
            for i in range(len(feature)):
                self.featureWeights[i] -= self.learningRate * feature[i] / featureSum * error
            