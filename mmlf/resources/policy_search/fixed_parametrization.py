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
""" Policy search method that optimizes the parameters of a fixed policy class

The module contains a policy search method that optimizes the parameters of 
an externally predefined, parametrized policy class. It can use an arbitrary 
black-box optimization algorithm for this purpose.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

from copy import deepcopy

from mmlf.resources.policy_search.policy_search import PolicySearch

class FixedParametrization(PolicySearch):
    """ Policy search method that optimizes the parameters of a policy
    
    The module contains a policy search method that optimizes the parameters of 
    an externally predefined policy.
    
    **CONFIG DICT**
        :policy: : A parametrized class of policies
        :optimizer: : An arbitrary black-box optimizer 
    """
    
    DEFAULT_CONFIG_DICT = {'policy':  {'type': 'linear', 'bias': True},
                           'optimizer' : {'name': 'evolution_strategy', 
                                          'evalsPerIndividual': 10, 
                                          'populationSize' : 5, 
                                          'numChildren' :10}}
    
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer
        self.initial_optimizer = deepcopy(optimizer) # required for resetting
            
    def getActivePolicy(self):
        """ Returns the currently active policy """
        parameters = self.optimizer.getActiveIndividual()
        policy = deepcopy(self.policy)
        policy.setParameters(parameters)
        return policy

    def getBestPolicy(self):
        """ Returns the best policy found so far """
        parameters = self.optimizer.getBestIndividual()
        policy = deepcopy(self.policy)
        policy.setParameters(parameters)
        return policy
    
    def getLastGenerationsBestPolicy(self):
        """ Returns the best policy of the last generation.
        
        Returns the best policy of the last generation that has been fully 
        evaluated. This method might be favorable in non-stationary environments
        that change over time.
        """
        parameters = self.optimizer.getLastGenerationsBestIndividual()
        policy = deepcopy(self.policy)
        policy.setParameters(parameters)
        return policy
    
    def tellFitness(self, fitnessSample):
        """ Gives one fitness sample for the currently active policy
        
        Provides the fitness *fitnessSample* obtained in one evaluation of
        the currently active policy.
        """
        self.optimizer.tellFitness(fitnessSample)
        
    def reinitialize(self):
        """ Reinitializes the policy search method """
        self.optimizer.reinitialize()
        
    def reset(self):
        """ Rest the policy search method """
        self.optimizer = deepcopy(self.initial_optimizer)
          
            