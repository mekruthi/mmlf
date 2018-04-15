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
""" Module for Eligibility Traces.

This module contains code for eligibility traces, which can be used in 
combination with temporal difference learning.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

from collections import defaultdict

class EligibilityTrace(object):
    """ A class that implements eligibility traces. """
    
    def __init__(self, minTraceValue = 0.01, replacingTraces = True):
        self.minTraceValue = minTraceValue
        self.replacingTraces = replacingTraces
        
        self.traces = defaultdict(float)
        
    def getTraces(self):
        """ Return a dictionary that stores all eligibilities."""
        return self.traces
        
    def setEligibility(self, state, action, eligibility):
        """ Set eligibility for this state action pair to the given value.
        
        Note: Uses replacing traces.
        """
        if self.replacingTraces:
            self.traces[(state, action)] = eligibility # Replacing traces!
        else:
            self.traces[(state, action)] += eligibility # Accumulating traces!
    
    def getEligibility(self, state, action):
        """ Return the eligibility of the given state action pair
         
        Returns 0.0 as default if the state action pair is unknown.
        """
        return self.traces[(state, action)]
            
    def decayAllEligibilities(self, decayRate):
        """ Decay all traces by *decayRate*. 
        
        Decay all traces by the given decayRate and remove state/action pairs 
        whose eligibility has decayed below the minTraceValue threshold
        """
        removeEntries = []
        # Decay all traces
        for state, action in self.traces.iterkeys():
            self.traces[(state, action)] *= decayRate
            # If eligibility is below threshold 
            if self.traces[(state, action)] < self.minTraceValue:
                # Remember it for removal
                removeEntries.append((state, action))
                
        # Remove eligibility traces that have decayed below minTraceValue    
        for state, action in removeEntries:
            self.traces.pop((state, action))
