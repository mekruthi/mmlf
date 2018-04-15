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
# Created: 2010/06/15
""" An action model class that is suited for discrete environments
    
This module contains a model that learns a distribution model for discrete
environments. 
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
import math
from collections import defaultdict

from mmlf.resources.model.model import ActionModel, ModelNotInitialized


class TabularModel(ActionModel):
    """ An action model class that is suited for discrete environments
        
    This module contains a model that learns a distribution model for discrete
    environments. 
    
    .. versionadded:: 0.9.9
    
    **CONFIG DICT**
        :exampleSetSize: : The maximum number of example transitions that is remembered in the example set. If the example set is full, old examples must be deleted or now new examples are accepted.
    """
    
    DEFAULT_CONFIG_DICT = {'exampleSetSize': 2500}
    
    SUPPORTED_STATE_SPACES = ["DISCRETE"]
    
    def __init__(self, stateSpace, **kwargs):
        super(TabularModel, self).__init__(stateSpace, **kwargs)
        
        self.stateTransitions = defaultdict(int)
        self.invStateTransitions = defaultdict(int)
        self.accumulatedReward = defaultdict(float)
        self.stateCounter = defaultdict(int)
        self.states = set()
    
    def addExperience(self, state, succState, reward):
        """ Add the given experience tuple to the example state transitions-
        
        Updates the model with one particular experience triple (s,s',r).
        (The action need not be specified since this model is 
        for one fixed action only).
        """
        self.stateTransitions[(state, succState)] += 1
        self.invStateTransitions[(succState, state)] += 1
        self.accumulatedReward[state] += reward
        self.stateCounter[state] +=1
        
        self.states.add(state)
        self.states.add(succState)
    
    def sampleState(self):
        """ Return a known state randomly sampled with uniform distribution"""
        if len(self.stateCounter.keys()) == 0:
            raise ModelNotInitialized()
        
        return random.choice(self.stateCounter.keys())
    
    def sampleSuccessorState(self, state):
        """ Return a states drawn from *state*'s successor distribution 
        
        Returns a possible successor state of *state* drawn from the successor 
        state distribution according to its probability mass function.
        """
        if self.stateCounter[state] == 0:
            raise ModelNotInitialized()
        
        computeProbabilityFct = \
            lambda succState: float(self.stateTransitions[(state, succState)]) \
                                    / self.stateCounter[state]
        probabilityMassFunction = [(succState, computeProbabilityFct(succState)) 
                                        for succState in self.states]
    
        randValue = random.random()
        accumulator = 0.0
        for succState, probabilityMass in probabilityMassFunction:
            accumulator += probabilityMass
            if accumulator >= randValue:
                return succState

        
    def getSuccessorDistribution(self, state):
        """ Return the successor distribution for the given *state*. 
        
        Returns an iterator that yields pairs of states and
        their probabilities of being the successor of the given *state*. 
        """
        if self.stateCounter[state] == 0:
            raise ModelNotInitialized()
        
        computeProbabilityFct = \
            lambda succState: float(self.stateTransitions[(state, succState)]) \
                                    / self.stateCounter[state]
        probabilityMassFunction = [(succState, computeProbabilityFct(succState)) 
                                        for succState in self.states]
        
        for succState, probabilityMass in probabilityMassFunction:
            if probabilityMass > 0.0:
                yield succState, probabilityMass
        

    def samplePredecessorState(self, state):
        """ Return a states drawn from *state*'s predecessor distribution 
        
        Returns a possible predecessor state of *state* drawn from the 
        predecessor state distribution according to its probability mass function.
        """
        if self.stateCounter[state] == 0:
            raise ModelNotInitialized()
        
        def computeProbabilityFct(predState):
            if self.stateCounter[predState] == 0:
                return 0.0
            else:
                return float(self.invStateTransitions[(state, predState)]) \
                                    / self.stateCounter[predState]
                                    
        probabilityMassFunction = [(predState, computeProbabilityFct(predState)) 
                                        for predState in self.states]
    
        randValue = random.random()
        accumulator = 0.0
        for predState, probabilityMass in probabilityMassFunction:
            accumulator += probabilityMass
            if accumulator >= randValue:
                return predState
        
    def getPredecessorDistribution(self, state):
        """ Return a states drawn from *state*'s predecessor distribution 
        
        Returns a possible predecessor state of *state* drawn from the 
        predecessor state distribution according to its probability mass function.
        """
        if self.stateCounter[state] == 0:
            raise ModelNotInitialized()
        
        def computeProbabilityFct(predState):
            if self.stateCounter[predState] == 0:
                return 0.0
            else:
                return float(self.invStateTransitions[(state, predState)]) \
                                    / self.stateCounter[predState]
                                    
        probabilityMassFunction = [(predState, computeProbabilityFct(predState)) 
                                        for predState in self.states]
        
        for predState, probabilityMass in probabilityMassFunction:
            if probabilityMass > 0.0:
                yield predState, probabilityMass
    
    def getExpectedReward(self, state):
        """ Returns the expected reward for the given state """
        if self.stateCounter[state] == 0:
            raise ModelNotInitialized()
        
        return self.accumulatedReward[state] / self.stateCounter[state]
    
    def getExplorationValue(self, state):
        """ Return the exploratory value of the given state *state*
        
        The exploratory value of a state under this model is defined simply as
        the sum of the activations of its k nearest neighbors 
        """
        return self.stateCounter[state]
    
    def plot(self, ax, stateSpace, plotStateDims, **kwargs):
        # Check some preconditions
        if stateSpace.getNumberOfDimensions() != 2:
            raise Exception("Plotting of tabular models works only for "
                            "two-dimensional state spaces.")
        try:
            import networkx as nx
        except ImportError:
            raise Exception("Plotting of tabular model requires the networkx "
                            "package.")
            
        # Construct a graph that encodes the state transition probabilities
        graph = nx.DiGraph()
        pos = {}
        labels = {}
        colors = {}
        
        for state in self.states:
            graph.add_node(state)
            pos[state] = (state[plotStateDims[0]], -state[plotStateDims[1]])
            labels[state] = self.stateCounter[state]
            #if isStartState(state):
            #    colors[state] = 'g'
            #elif isTerminalState(state):
            #    colors[state] = 'r'
            #else:
            #    colors[state] = 'b'
            colors[state] = 'b'
            try:
                for succState, probMass in self.getSuccessorDistribution(state):
                    graph.add_edge(succState, state, weight = probMass)
            except ModelNotInitialized:
                continue 
                
        # Draw this graph
        for u,v,d in graph.edges(data=True):
            nx.draw_networkx_edges(graph, pos, edgelist=[(u,v)], ax=ax, 
                                   alpha=math.sqrt(d['weight']), edge_color='k',
                                   width=2.0, with_labels=False)
        
        nx.draw(G=graph, pos=pos, with_labels=False,
                node_color=[colors[state] for state in self.states])
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels)
        