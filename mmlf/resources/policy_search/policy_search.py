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
# Created: 2009/11/25
""" Interface for MMLF policy search methods

MMLF policy search methods must implement the following methods:
 * getActivePolicy 
 * getBestPolicy 
 * tellFitness
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from mmlf.resources.policies.policy import Policy
from mmlf.resources.optimization.optimizer import Optimizer

class PolicySearch(object):
    """ Interface for MMLF policy search methods

    MMLF optimizers must implement the following methods:
     * getActivePolicy(): Returns the currently active policy
     * getBestPolicy(): Returns the best policy found so far
     * tellFitness(fitnessSample): Gives one fitness sample for the currently 
                                   active policy
    """
    
    # A dictionary that will contain a mapping from policy search name
    # to the respective policy search class
    PS_DICT= None
            
    def getActivePolicy(self):
        """ Returns the currently active policy """
        raise NotImplementedError("PolicySearch is only an interface")

    def getBestPolicy(self):
        """ Returns the best policy found so far """
        raise NotImplementedError("PolicySearch is only an interface")
    
    def tellFitness(self, fitnessSample):
        """ Gives one fitness sample for the currently active policy
        
        Provides the fitness *fitnessSample* obtained in one evaluation of
        the currently active policy.
        """
        raise NotImplementedError("PolicySearch is only an interface")
    
    def reset(self):
        """ Resets the policy search method """
        raise NotImplementedError("PolicySearch is only an interface")

    @staticmethod
    def getPolicySearchDict():
        """ Returns dict that contains a mapping from PS name to PS class. """
        # Lazy initialization  
        if PolicySearch.PS_DICT == None:
            from mmlf.resources.policy_search.fixed_parametrization \
                        import FixedParametrization 
            
            # A static dictionary containing a mapping from name to FA class 
            PolicySearch.PS_DICT = {'fixed_parametrization': FixedParametrization}
            
        return PolicySearch.PS_DICT

    @staticmethod
    def create(policySearchSpec, stateSpace, actionSpace, stateDims):
        """ Factory method that creates policy search method based on spec. """
        ps_dict = PolicySearch.getPolicySearchDict()
        if policySearchSpec["method"] == "fixed_parametrization":
            # Lazy initialization of the policy
            policy = Policy.create(policySearchSpec["policy"], stateDims,
                                   actionSpace)   

            # Creating the optimizer
            optimizer = Optimizer.create(policySearchSpec["optimizer"],
                                         sampleInstance = policy.getParameterSampleFunction())
            
            # Update the policies parameters
            parameters = optimizer.getActiveIndividual()
            policy.setParameters(parameters)
            
            # Create policy search method
            policySearch = ps_dict["fixed_parametrization"](policy, optimizer)
        else:
            raise NotImplementedError("Policy search method %s unknown" % policySearchSpec["method"])
        
        # Remember original spec file
        policySearch.policySearchSpec = policySearchSpec
        
        return policySearch
