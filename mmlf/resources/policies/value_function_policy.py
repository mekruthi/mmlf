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
# Created: 2008/06/27
""" Policies that are represented using value function 

This module contains a class that wraps function approximators such that
they can be used directly as policies (i.e. implement the policy interface) 
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

from mmlf.resources.policies.policy import Policy 

class ValueFunctionPolicy(Policy):
    """ Class for policies that are represented using value function 
    
    This class wraps a function approximator such that it can be used directly
    as policy (i.e. implement the policy interface) 
    """
    def __init__(self, valueFunction, actions):
        self.valueFunction = valueFunction
        self.actions = actions      

    def evaluate(self, state):
        """ Evaluates the policy for the given state """
        # Compute the optimal action
        optimalAction = max([(self._getStateActionValue(state, action),
                                action) for action in self.actions])[1]
        
        return optimalAction
    
    def getOptimalActions(self, state):
        """ Returns all actions that would be optimal for the given state. """
                # Compute the optimal action
        optimalActionValue = max(self._getStateActionValue(state, action)   
                                            for action in self.actions)
        return [action for action in self.actions 
                    if self._getStateActionValue(state, action) == optimalActionValue]

        
    def getParameters(self):
        """ Returns the parameters of this policy """
        return [] # TODO
    
    def setParameters(self, parameters):
        """ Sets the parameters of the policy to the given parameters """
        pass # TODO

    def _getStateActionValue(self, state, action):
        """
        Compute the Q value for the given state-action pair
        """
        return self.valueFunction.computeQ(state, action)
    