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

# Author: Jan Hendrik Metzen  (jhm@informatik.uni-bremen.de)
# Created: 2010/05/28

""" Module that contains the seventeen & four environment

This module contains a simplified implementation of the card game seventeen & four,
in which the agent takes the role of the player and plays against a hard-coded 
dealer.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random

from mmlf.framework.spaces import StateSpace, ActionSpace
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.environments.single_agent_environment import SingleAgentEnvironment
from mmlf.framework.observables import FloatStreamObservable

class SeventeenAndFourEnvironment(SingleAgentEnvironment):
    """ The seventeen & four environment
    
    This environment implements a simplified form of the card game seventeen & four,
    in which the agent takes the role of the player and plays against a hard-coded 
    dealer.
    
    The player starts initially with two randomly drawn card with values of
    2,3,4,7,8,9,10 or 11. The goal is get a set of cards whose sum is as close 
    as possible to 21. The agent can stick with two cards or draw arbitrarily 
    many cards sequentially. If the sum of cards becomes greater than 21, the
    agent looses and gets a reward of -1. If the agent stops with cards less 
    valued than 22, a hard-coded dealer policy starts playing against the agent.
    This dealer draws card until it has either equal/more points than the agent 
    or more than 21. In the first case, the dealer wins and the agent gets a 
    reward of -1, otherwise the player wins and gets a reward of 0.   
    """
    
    DEFAULT_CONFIG_DICT = {}
    
    def __init__(self, useGUI, *args, **kwargs):
        
        self.environmentInfo = EnvironmentInfo(versionNumber="0.3",
                                               environmentName="17 and 4",
                                               discreteActionSpace=True,
                                               episodic=True,
                                               continuousStateSpace=False,
                                               continuousActionSpace=False,
                                               stochastic=True)
        
        super(SeventeenAndFourEnvironment, self).__init__(useGUI=useGUI, *args, **kwargs) 
        
        # State and action space definition
        oldStyleStateSpace =   {"count": ("discrete", range(23))}

        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")

        oldStyleActionSpace =  {"action": ("discrete", ["continue", "stop"])}

        self.actionSpace = ActionSpace()
        self.actionSpace.addOldStyleSpace(oldStyleActionSpace, limitType="hard") 
        
        # The available cards
        self.cards = [2,2,2,2,3,3,3,3,4,4,4,4,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,
                      10,11,11,11,11]
        # Initialize first game
        self.getInitialState()  
        
        # Some observables
        self.pointsObservable = \
                FloatStreamObservable(title='%s Points' % self.__class__.__name__,
                                      time_dimension_name='Episode',
                                      value_name='Points')
        
        if useGUI:
            from mmlf.gui.viewers import VIEWERS
            from mmlf.worlds.seventeen_and_four.environments.seventeen_and_four_viewers\
                        import SeventeenAndFourValuefunctionViewer
            # Add a Q-value viewer for this world
            VIEWERS.addViewer(lambda : SeventeenAndFourValuefunctionViewer(self.stateSpace),
                              'SeventeenAndFourValuefunctionViewer')
    
    ########################## Interface Functions #####################################   
    def getInitialState(self):
        """ Returns the initial state of the environment """
        self.remainingCards = list(self.cards)
        self.drawnCards = []
        # Player starts with two cards
        self._drawCard(self.drawnCards)
        self._drawCard(self.drawnCards)
        return self._createState()
    
    def evaluateAction(self, actionObject):
        """ Execute an agent's action in the environment.
        
        Take an actionObject containing the action of an agent, and evaluate 
        this action, calculating the next state, and the reward the agent 
        should receive for having taken this action.
        
        Additionally, decide whether the episode should continue,
        or end after the reward has been  issued to the agent.
        
        This method returns a dictionary with the following keys:
           :rewardValue: : An integer or float representing the agent's reward.
                           If rewardValue == None, then no reward is given to the agent.
           :startNewEpisode: : True if the agent's action has caused an episode
                               to get finished.
           :nextState: : A State object which contains the state the environment
                         takes on after executing the action. This might be the
                         initial state of the next episode if a new episode
                         has just started (startNewEpisode == True)
           :terminalState: : A State object which contains the terminal state 
                             of the environment in the last episode if a new 
                             episode has just started (startNewEpisode == True). 
                             Otherwise None.        
        """
        if actionObject['action'] == 'stop':
            # Agent stopped
            self.episodeCounter += 1
            sumOfCards = sum(self.drawnCards) 
            self.pointsObservable.addValue(self.episodeCounter, sumOfCards)
            # Determine dealers outcome
            dealersCards = []
            self._drawCard(dealersCards) # Dealer starts with two cards
            self._drawCard(dealersCards)

            # Dealer draws until he has same number of points as agent or too many
            while sum(dealersCards) <= sumOfCards and sum(dealersCards) < 22:
                self._drawCard(dealersCards)
            
            self.environmentLog.info("Episode %s: Agent %s Dealer %s" 
                                       % (self.episodeCounter, sumOfCards, 
                                          sum(dealersCards)))
            if sum(dealersCards) > sumOfCards and sum(dealersCards) < 22:
                # Agent lost against dealer
                self.returnObservable.addValue(self.episodeCounter,
                                                -1)
                return {"reward" : -1, # lost
                        "terminalState" : {'count': 22},
                        "nextState" : self.getInitialState(),
                        "startNewEpisode" : True}
            else:
                # Agent won since it has more points than dealer
                self.returnObservable.addValue(self.episodeCounter, 0)
                return {"reward" : 0, # won
                        "terminalState" : {'count': 22},
                        "nextState" :  self.getInitialState(),
                        "startNewEpisode" : True}
                
        # Draw a card
        self._drawCard(self.drawnCards)
        
        if sum(self.drawnCards) > 21:
            # Agent lost since its cards exceed 21
            self.environmentLog.info("Episode %s: Agent %s"
                                      % (self.episodeCounter, sum(self.drawnCards)))
            self.episodeCounter += 1
            self.pointsObservable.addValue(self.episodeCounter,
                                           sum(self.drawnCards))
            self.returnObservable.addValue(self.episodeCounter, -1)
            return {"reward" : -1, # lost
                    "terminalState" : {'count': 22},                    
                    "nextState" : self.getInitialState(),
                    "startNewEpisode" : True}
        else:
            return {"reward" : 0, # game still running
                    "terminalState" : None,
                    "nextState" : self._createState(),
                    "startNewEpisode" : False}
        
    def _createState(self):
        return {'count': min(22, sum(self.drawnCards))}
    
    def _drawCard(self, listOfCards):
        """ Draw a card randomly """
        card = random.choice(self.remainingCards)
        self.remainingCards.remove(card)
        listOfCards.append(card)
        

EnvironmentClass = SeventeenAndFourEnvironment
EnvironmentName = "Seventeen and Four"

