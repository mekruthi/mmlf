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
# Created: 2010/11/25
""" MMLF skill discovery interface

This module defines the interface for skill discovery methods

The following  methods must be implemented by each skill discovery method:
 * *tellStateTransition(self, state, action, reward, succState)*:
      Inform skill discovery of a new transition and search for new skills.         
 * *episodeFinished()*: 
      Inform skill discovery that an episode has terminated.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


class SkillDiscovery(object):
    """ The skill discovery interface. 
    
    .. versionadded:: 0.9.9
    """
    
    # A dictionary that will contain a mapping from skill discovery name
    # to the respective skill discovery class
    SKILL_DISCOVERY_DICT= None
    
    # Default configuration of this skill discovery method
    DEFAULT_CONFIG_DICT = {}
    
    def __init__(self, optionCreationFct, stateSpace, *args, **kwargs):
        self.optionCreationFct = optionCreationFct
        self.stateSpace = stateSpace
        
        # The set of terminal states encountered in the environment
        self.terminalStates = set()
        
        # The set of identified options
        self.options = set()
    
    def tellStateTransition(self, state, action, reward, succState):
        """ Inform skill discovery method about a new state transition.
        
        The agent has transitioned from *state* to *succState* after executing 
        *action* and obtaining the reward *reward*.
        
        Returns a pair consisting of the list of discovered options and whether
        a new option was discovered based on this state transition.
        """
        return [], False
    
    def episodeFinished(self, terminalState):
        """ Inform skill discovery method that the current episode terminated."""
        # Create an option for every terminal state for reaching it from 
        # its neighborhood
        addedOption = False
#        if terminalState not in self.terminalStates:
#            self.terminalStates.add(terminalState)
#            addedOption = True
#            terminalNode = (terminalState['column'], terminalState['row'])
#            distanceFct = \
#                lambda state: abs(state['column'] - terminalState['column']) \
#                                + abs(state['row'] - terminalState['row'])
#            reachTerminalStateOption = \
#                    self.optionCreationFct(subgoal=terminalNode,
#                                           distanceFct=distanceFct)
#            self.options.add(reachTerminalStateOption)
            
        return self.options, addedOption
    
    @staticmethod
    def getSkillDiscoveryDict():
        """ Returns dict that contains a mapping from skill discovery name to class. """
        # Lazy initialization  
        if SkillDiscovery.SKILL_DISCOVERY_DICT == None:
            # A static dictionary containing a mapping from name to class 
            SkillDiscovery.SKILL_DISCOVERY_DICT = \
                            {'no_skill_discovery': SkillDiscovery}
            try:
                from mmlf.resources.skill_discovery.graph_partitioning \
                                            import LocalGraphPartitioning
                SkillDiscovery.SKILL_DISCOVERY_DICT['graph_partitioning'] \
                                       = LocalGraphPartitioning
            except ImportError:
                pass
            try: 
                from mmlf.resources.skill_discovery.predefined_skills \
                                            import PredefinedSkills
                SkillDiscovery.SKILL_DISCOVERY_DICT['predefined_skills'] \
                                      = PredefinedSkills
            except ImportError:
                pass
            
        return SkillDiscovery.SKILL_DISCOVERY_DICT

    @staticmethod
    def create(skillDiscoverySpec, optionCreationFct, stateSpace):
        """ Factory method that creates skill discovery based on spec-dictionary. """
        # Determine the model class
        if skillDiscoverySpec['name'] in SkillDiscovery.getSkillDiscoveryDict():
            SkillDiscoveryClass = \
                    SkillDiscovery.getSkillDiscoveryDict()[skillDiscoverySpec['name']]
        else:
            raise NotImplementedError("Skill Discovery type %s unknown" 
                                      % skillDiscoverySpec["name"])
        
        return SkillDiscoveryClass(optionCreationFct, stateSpace,
                                   **skillDiscoverySpec)
