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
# Created: 2009/11/23
""" Interface for MMLF policies

A policy is a mapping from state to action. Different learning algorithms of 
the MMLF have different representation of policies, for example neural networks
which represents the policy directly or polcies based on value functions.

This module encapsulates these details so that a stored policy can be 
loaded directly.

MMLF policies must implement the following methods:
 * evaluate 
 * getParameters 
 * setParameters
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

class Policy(object):
    """ Interface for MMLF policies
    
    MMLF policies must implement the following methods:
     * evaluate(state): Evaluates the deterministic policy for the given state 
     * getParameters(): Returns the parameters of this policy
     * setParameters(parameters): Sets the parameters of the policy to the given parameters
    """
    
    # A dictionary that will contain a mapping from policy name
    # to the respective policy class
    POLICY_DICT= None
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Policy is only an interface")
        
    def evaluate(self, state):
        """ Evaluates the deterministic policy for the given state """
        raise NotImplementedError("Policy is only an interface")
    
    def getParameters(self):
        """ Returns the parameters of this policy """
        raise NotImplementedError("Policy is only an interface")
    
    def setParameters(self, parameters):
        """ Sets the parameters of the policy to the given parameters """
        raise NotImplementedError()

    @staticmethod
    def getPolicyDict():
        """ Returns dict that contains a mapping from policy name to policy class. """
        # Lazy initialization  
        if Policy.POLICY_DICT == None:
            from mmlf.resources.policies.mlp_policy import MLPPolicy 
            from mmlf.resources.policies.linear_policy \
                    import LinearDiscreteActionPolicy
            
            # A static dictionary containing a mapping from name to class 
            Policy.POLICY_DICT = {'linear': LinearDiscreteActionPolicy, # Handled differently
                                  'mlp': MLPPolicy}
            
        return Policy.POLICY_DICT

    @staticmethod
    def create(policySpec, numStateDims, actionSpace):
        """ Factory method that creates policy based on spec-dictionary. """
        # Lazy initialization of the policy
        if policySpec["type"] == 'linear':
            if actionSpace.hasContinuousDimensions():
                from mmlf.resources.policies.linear_policy \
                    import LinearContinuousActionPolicy
                return  LinearContinuousActionPolicy(numStateDims,
                                                     actionSpace,
                                                     **policySpec)
            else: # Discrete action spaces
                from mmlf.resources.policies.linear_policy \
                    import LinearDiscreteActionPolicy
                return LinearDiscreteActionPolicy(numStateDims, 
                                                  actionSpace,
                                                  **policySpec)
        elif policySpec["type"] == 'mlp':
            return Policy.getPolicyDict()['mlp'](inputDims=numStateDims, 
                                                 actionSpace=actionSpace,
                                                 **policySpec)

        raise NotImplementedError("Policy type %s unknown" % policySpec["type"])
           