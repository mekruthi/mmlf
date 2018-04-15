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
""" MMLF function approximator interface

This module defines the interface for function approximators that can be  
used with temporal difference learning methods.

The following  methods must be implemented by each function approximator:
 * *computeQ(state, action)*: Compute the Q value of the given state-action pair 
 * *train()*: Train the function approximator on the trainSet consisting of
              state-action pairs and the desired Q-value for these pairs.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random 

class InappropriateFunctionApproximatorException(Exception):
    pass

class FunctionApproximator(object):
    """ The function approximator interface. 
    
    Each function approximator must specify two methods:
    * computeQ
    * train
    """
    
    # A dictionary that will contain a mapping from function approximator name
    # to the respective function approximator class
    FA_DICT= None
    
    def __init__(self, stateSpace, *args, **kwargs):
        if stateSpace and stateSpace.hasContinuousDimensions() \
                and "CONTINUOUS" not in self.__class__.SUPPORTED_STATE_SPACES:
            exceptionString = "Function approximator %s does not support " \
                              "continuous state spaces." % self.__class__.__name__
            raise InappropriateFunctionApproximatorException(exceptionString)
        elif stateSpace and not stateSpace.hasContinuousDimensions() \
                and "DISCRETE" not in self.__class__.SUPPORTED_STATE_SPACES:
            exceptionString = "Function approximator %s does not support " \
                              "discrete state spaces." % self.__class__.__name__
            raise InappropriateFunctionApproximatorException(exceptionString)
    
    @staticmethod
    def getFunctionApproximatorDict():
        """ Returns dict that contains a mapping from FA name to FA class. """
        # Lazy initialization  
        if FunctionApproximator.FA_DICT == None:
            from mmlf.resources.function_approximators.cmac import CMAC
            from mmlf.resources.function_approximators.multilinear_grid \
                                    import MultilinearGrid
            from mmlf.resources.function_approximators.rbf import RBF_FA
            from mmlf.resources.function_approximators.mlp import MLP
            from mmlf.resources.function_approximators.tabular_storage \
                                    import TabularStorage
            from mmlf.resources.function_approximators.linear_combination \
                                    import LinearCombination
            from mmlf.resources.function_approximators.meta import MultiFunctions
                
            # A static dictionary containing a mapping from name to FA class 
            FunctionApproximator.FA_DICT = {'CMAC': CMAC,
                                            'MultilinearGrid': MultilinearGrid,
                                            'RBF': RBF_FA,
                                            'MLP': MLP,
                                            'TabularStorage': TabularStorage,
                                            'MultiFunctions': MultiFunctions}
            
            try: # Depends on scikits.ann
                from mmlf.resources.function_approximators.knn \
                                        import KNNFunctionApproximator
                FunctionApproximator.FA_DICT['KNN'] = KNNFunctionApproximator
            except ImportError, e:
                pass
            
        return FunctionApproximator.FA_DICT
    
    @staticmethod
    def create(faSpec, stateSpace, actions):
        """ Factory method that creates function approximator based on spec-dictionary. """
        # Determine the function approximator class
        if faSpec['name'] in FunctionApproximator.getFunctionApproximatorDict():
            FuncApproxCls = FunctionApproximator.getFunctionApproximatorDict()[faSpec['name']]
        else:
            raise Exception("No function approximator %s known." % faSpec['name'])
        
        #Create a function approximator
        functionApproximator = FuncApproxCls(stateSpace=stateSpace,
                                             actions=actions,
                                             **faSpec)
        
        return functionApproximator
    
    def computeOptimalAction(self, state):
        """ Compute the action with maximal Q-value for the given state """
        actionValues = [(self.computeQ(state, action),
                         random.random(), # To randomize choice of equally valued action
                         action) for action in self.actions]
        return max(actionValues)[2]
    
    def computeV(self, state):
        """ Computes the V-value of the given state"""
        return self.computeQ(state, self.computeOptimalAction(state))
    
    def computeQ(self, state, action):
        """ Computes the Q-value of the given state, action pair
        
        It is assumed that a state is a n-dimensional vector 
        where n is the dimensionality of the state space. Furthmore, the
        states must have been scaled externally so that the value of each dimension
        falls into the bin [0,1]. action must be one of the actions given to the constructor
        """
        raise NotImplementedError('The abstract class FunctionApproximator cannot be used directly') 
    
    def train(self, trainingSet):
        """ Trains the function approximator using the given training set.
        
        trainingSet is a dictionary containing training data where the 
        key is the respective (state, action) pair whose Q-value should
        be updated and the dict-value is this desired Q-value.
        """
        raise NotImplementedError('The abstract class FunctionApproximator cannot be used directly')
        