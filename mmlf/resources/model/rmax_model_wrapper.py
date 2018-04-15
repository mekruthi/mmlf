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
# Created: 2010/07/14
""" A wrapper for models that changes them to have RMax like behavior
    
A wrapper that wraps a given *model*  and changes its behavior to be 
RMax-like, i.e. return *RMax* instead of the reward predicted by
*model* if the exploration value is below *minExplorationValue*.
The implementation is based on the adapter pattern.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


from mmlf.resources.model.model import Model, ModelNotInitialized

class RMaxModelWrapper(Model):
    """ A wrapper for models that changes them to have RMax like behavior
    
    A wrapper that wraps a given *model*  and changes its behavior to be 
    RMax-like, i.e. return *RMax* instead of the reward predicted by
    *model* if the exploration value is below *minExplorationValue*.
        
    .. versionadded:: 0.9.9
        
    **CONFIG DICT** 
        :minExplorationValue: : The agent explores in a state until the given exploration value (approx. number of exploratory actions in proximity of state action pair) is reached for all actions
        :RMax: : An upper bound on the achievable return an agent can obtain in a single episode
        :model: The actual model (this is only a wrapper arount the true model that implements optimism in the face of uncertainty)
    """
    
    DEFAULT_CONFIG_DICT = {'model' : {'name': 'KNNModel',
                                      'k': 100,
                                      'b_Sa': 0.03,
                                      'exampleSetSize': 2500},
                           'RMax' : 0.0,
                           'minExplorationValue' : 1.0}
    
    def __init__(self, model, RMax, minExplorationValue, **kwargs):
        self.model = model
        self.RMax = RMax
        self.minExplorationValue = minExplorationValue
        
        # For all methods and attributes except the explicitly defined ones, 
        # the wrapper should just delegate to the wrapped model 
        for item in dir(model):
            if item.startswith("__") or item in dir(self): continue
            self.__dict__[item] = eval("model.%s" % item)
        
        
    def getExpectedReward(self, state, action):
        """ Reward expectation when using RMax-like reward modification. 
        
        Returns Rmax if the exploration value of the given *state*-*action* pair
        is below minExplorationValue, otherwise returns the reward predicted by
        the wrapped model
        """
        try:
            if self.model.getExplorationValue(state, action) < self.minExplorationValue:
                return self.RMax
            else:
                return self.model.getExpectedReward(state, action)
        except ModelNotInitialized:
            # If the model is not initialized, we also return RMax
            return self.RMax
    