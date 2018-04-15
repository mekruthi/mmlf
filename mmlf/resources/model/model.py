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
# Created: 2009/02/23
""" Interfaces for MMLF models

This module contains the Model class that specifies the interface for models
in the MMLF. The following  methods must be implemented by each model class:
 * addExperience
 * sampleStateAction
 * sampleSuccessorState
 * getSuccessorDistribution
 * getExpectedReward
 * getExplorationValue
 
The standard way of implementing a model is to learn a model for each
action separately. In order to simplify this task, the Model interface 
contains a standard implementation in the case that an ActionModelClass
parameter is passed to constructor. This parameters must be a class that
implements the interface ActionModel that is also contained in this 
module. If the ActionModelClass parameter is passed to the Model
constructor, for each action one instance of it is constructed
an all methods are per default forwarded to the respective method
of the ActionModel.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
import itertools
import numpy
import copy

from mmlf.framework.state import State
from mmlf.framework.spaces import Dimension
from mmlf.framework.observables import ModelObservable

class ModelNotInitialized(Exception): 
    """ An exception indicating that the model is not initialized """
    pass

class Model(object):
    """ Interface for MMLF models
    
    This class specifies the interface for models
    in the MMLF. The following  methods must be implemented by each model:
     * addExperience
     * getSample
     * sampleSuccessorState
     * getSuccessorDistribution (optional)
     * getExpectedReward
     * getExplorationValue (optional)
     
    Additional, this class already implements the following methods:
     * addStartState
     * addTerminalState
     * drawStartState
     * getNearestNeighbor
     * getNearestNeighbors
     * isTerminalState
     * plot
     
    The class contains a standard implementation in the case that an 
    ActionModelClass parameter is passed to constructor.
    This parameters must be a class that implements the interface ActionModel
    that is also contained in this module. 
    If the ActionModelClass parameter is passed to the Model
    constructor, for each action one instance of it is constructed
    an all methods are per default forwarded to the respective method
    of the ActionModel.
    
    The constructor expects the following parameters:
     * *agent* : The agent that is using this model
     * *userDirObj* : The userDirObj object of the agent
     * *actions* : A sequence of all allowed actions 
                  (note: discrete action space assumed)  
     * *ActionModelClass* : A class implementing the ActionModel interface
                            If specified, for each available action a separate
                            instance is created and all method calls are 
                            forwarded to the respective instance
     * *cyclicDimensions* If some (or all) of the dimensions of the state space 
                          are cyclic (i.e. the value 1.0 and 0.0 are equivalent)
                          this parameter should be set to True. Defaults to
                          False.
    """
    
    # A dictionary that will contain a mapping from model name
    # to the respective model class
    MODEL_DICT= None
    
    def __init__(self, agent, userDirObj, stateSpace, actions, 
                 ActionModelClass=None, cyclicDimensions=False, **kwargs):
        self.userDirObj = userDirObj
        self.actions = actions
        self.cyclicDimensions = cyclicDimensions
        
        # Late import to avoid cyclic dependencies
        from mmlf.resources.model.example_set import ExampleSet
        
        # Data structures to remember all visited states and whether 
        # they are start or terminal states 
        self.exampleSet = ExampleSet(kwargs['exampleSetSize'])
        self.startStates = set()
        self.terminalStates = ExampleSet(kwargs['exampleSetSize'] / 10)
            
        # If ActionModelClass parameters is passed
        if ActionModelClass != None:
            # Per default, a separate model is learned for each dimension
            self.actionModels = \
                dict([(action, ActionModelClass(stateSpace, **kwargs))
                                    for action in self.actions])
                
        # Create model observable if problem is continuous
        self.modelObservable = None
        if not stateSpace.hasDiscreteDimensions():
            self.modelObservable = \
                    ModelObservable('%s (model)' % agent.__class__.__name__)
    
    @staticmethod
    def getModelDict():
        """ Returns dict that contains a mapping from model name to model class. """
        # Lazy initialization  
        if Model.MODEL_DICT == None:
            # A static dictionary containing a mapping from name to class 
            Model.MODEL_DICT = {}
            try:
                from mmlf.resources.model.example_set import ExampleSet
                try:
                    from mmlf.resources.model.tabular_model import TabularModel
                    Model.MODEL_DICT['TabularModel'] = TabularModel
                except ImportError:
                    pass
                try:
                    from mmlf.resources.model.grid_model import GridModel
                    Model.MODEL_DICT['GridModel'] = GridModel
                except ImportError:
                    pass
                try:
                    from mmlf.resources.model.knn_model import KNNModel
                    Model.MODEL_DICT['KNNModel'] = KNNModel
                except ImportError:
                    pass
                try:
                    from mmlf.resources.model.rmax_model_wrapper import RMaxModelWrapper
                    Model.MODEL_DICT['RMaxModelWrapper'] = RMaxModelWrapper
                except ImportError:
                    pass
                try:
                    from mmlf.resources.model.lwpr_model import LWPRModel
                    Model.MODEL_DICT['LWPRModel'] = LWPRModel
                except ImportError:
                    pass
            except ImportError:
                pass
            
            if Model.MODEL_DICT == {}: # We can not use model-based learning
                import mmlf
                mmlf.log.warning("There are no models that are usable on your system. "
                                 "Probably, you lack scikits.ann. We remove all "
                                 "model-based agents.")
                from mmlf.agents import MMLF_AGENTS
                MMLF_AGENTS.pop("Dyna TD", None)
                MMLF_AGENTS.pop("Model-based Direct Policy Search", None)
                MMLF_AGENTS.pop("Fitted-RMax", None)
        
        return Model.MODEL_DICT
    
    @staticmethod
    def create(modelSpec, agent, stateSpace, actionSpace, userDirObj):
        """ Factory method that creates model-learners based on spec-dictionary. """
        
        # Determine the model class
        if modelSpec['name'] in Model.getModelDict():
            ActionModelClass = Model.getModelDict()[modelSpec['name']]
        else:
            raise Exception("No model with name %s known." % modelSpec['name'])
    
        # Create model
        if actionSpace.hasContinuousDimensions():
            assert actionSpace.getNumberOfDimensions() == 1, \
                "Models can currently not deal with continuous action spaces"\
                "with more than one dimension!"
            actionDimension = actionSpace.getDimensions()[0] # there is per assert only 1
            actionRanges = actionDimension.getValueRanges()
            assert len(actionRanges) == 1, "Models cannot deal with "\
                                           "non-contiguous action ranges."
            actionRange = actionRanges[0] # there is per assert only 1
            
            # If the action space has continuous dimensions, we treat them
            # as states in the model
            return JointStateActionModel(agent=agent,actionRange=actionRange,
                                         actionSpace=actionSpace,
                                         ActionModelClass=ActionModelClass,
                                         **modelSpec)
        elif modelSpec["name"] == "RMaxModelWrapper":
            from mmlf.resources.model.rmax_model_wrapper import RMaxModelWrapper
            # The encapsulated model
            model = Model.create(modelSpec["model"], agent, stateSpace, 
                                 actionSpace, userDirObj)
            modelSpec.pop("model")
            # wrap that model by an rmax optimistic wrapper
            return RMaxModelWrapper(model, **modelSpec)
        else:
            actions = actionSpace.getActionList()
            # We learn the model for each discrete dimension separately
            return Model(agent=agent, userDirObj=userDirObj, stateSpace=stateSpace,
                         actions=actions, ActionModelClass=ActionModelClass,
                         **modelSpec)
    
    def addStartState(self, state):
        """ Add the given state to the set of start states """
        self.startStates.add(state)
        # Inform observable of changed model
        if self.modelObservable:
            self.modelObservable.updateModel(self)

    def addExperience(self, state, action, succState, reward):
        """ Updates the model based on the given experience tuple
        
        Update the model based on the given experience tuple consisting
        of a *state*, an *action* taken in this state, the resulting
        successor state *succState* and the obtained *reward*.
        """
        self.exampleSet.addTransition(state, succState, reward)

        # If some of the dimensions are cyclic, we check if the value of one
        # dimension has changed by more than 0.5. If yes, we assume that we 
        # have moved in the opposite direction, e.g. instead of 0.9 -> 0.1 from
        # 0.9 to 1.1. This makes model learning easier.
        if self.cyclicDimensions:
            displacementVector = succState - state 
            if numpy.any(displacementVector > 0.5) or numpy.any(displacementVector < -0.5):
                print "Cyclic displacement from %s to %s" % (state, succState) 
            succState[numpy.where(displacementVector > 0.5)] -= 1.0
            succState[numpy.where(displacementVector < -0.5)] += 1.0

        returnValue = None
        if hasattr(self, "actionModels"):
            returnValue = \
                self.actionModels[action].addExperience(state, succState, reward)
                
        # Inform observable of changed model
        if self.modelObservable:
            self.modelObservable.updateModel(self)
            
        return returnValue
    
    def addTerminalState(self, state):
        """ Add the given state to the set of terminal states """
        self.terminalStates.addTransition(state, state, 0)
        
        if self.modelObservable:
            self.modelObservable.updateModel(self)
        
    def getStates(self):
        """ Return all states that are contained in the example set or are terminal"""
        states = self.exampleSet.getStates()
        if not self.terminalStates.isEmpty():
            states.extend(self.terminalStates.getStates())
        return states

    def drawStartState(self):
        """ Returns a random start state """
        return random.choice(list(self.startStates))
    
    def getSample(self):
        """ Return a sample drawn randomly 
        
        Return a random sample (i.e. a state, action, reward, successor
        state 4-tuple).
        """
        if hasattr(self, "actionModels"):
            try: 
                action = random.choice(self.actions)
                state = self.actionModels[action].sampleState()
                reward = self.getExpectedReward(state, action)
                succState = self.sampleSuccessorState(state, action)
                if succState == None:
                    raise ModelNotInitialized("Not enough experience to sample from model")                    
                return (state, action, reward, succState)
            except IndexError:
                raise ModelNotInitialized("Not enough experience to sample from model")
        else:
            raise NotImplementedError()
    
    def getNearestNeighbor(self, state):
        """ Returns the most similar known state to the given *state* """
        return list(self.getNearestNeighbors(state, 1, b=1))[0][1] # b doesn't matter
        
    def getNearestNeighbors(self, state, k, b):
        """  Determines *k* most similar states to the given *state*
        
        Determines *k* most similar states to the given *state*. Returns an
        iterator over (weight, neighbor), where weight is the guassian weigthed
        influence of the neighbor onto *state*. The weight is computed via
        exp(-dist/b**2)/sum_over_neighbors(exp(-dist_1/b**2)).
        Note that the weights sum to 1.
        """
        # The nearest k non-terminal state 
        nearestNeighborsNT = \
            list(self.exampleSet.getNearestNeighbors(state, k, b))
        # The nearest k terminal state 
        if not self.terminalStates.isEmpty():
            nearestNeighborsT = \
                list(self.terminalStates.getNearestNeighbors(state, k, b))
        else:
            nearestNeighborsT = []
        
        # return iterator over the k nearest neighbors
        nearestNeighbors = list(itertools.chain(nearestNeighborsNT, 
                                                nearestNeighborsT))[:k]
        return nearestNeighbors
        
    def isTerminalState(self, state):
        """ Returns an estimate of whether the given state is a terminal one """
        # If no terminal states are known, we estimate this state to be
        # non-terminal, too
        if self.terminalStates.isEmpty():
            return False
        
        # Determine nearest known non-terminal state and nearest known 
        # terminal state
        nearestNeighbor = self.exampleSet.getNearestNeighbor(state)
        nearestTerminalState = self.terminalStates.getNearestNeighbor(state)
        # If the state is closer to the nearest terminal state than to the
        # nearest non-terminal state, it is assumed to be terminal itself. 
        distNN = numpy.linalg.norm(state - nearestNeighbor)
        distTS = numpy.linalg.norm(state - nearestTerminalState) 
        return distTS <= distNN
      
    def sampleSuccessorState(self, state, action):
        """ Return sample successor state of state-action.
        
        Return a state drawn randomly from the successor distribution 
        of *state*-*action*
        """
        if hasattr(self, "actionModels"):
            succState = self.actionModels[action].sampleSuccessorState(state)
            # We compensate for the modifications we have made for cyclic
            # dimensions.
            if self.cyclicDimensions:                    
                succState = numpy.array(succState)
                succState[succState > 1.0] -= 1.0
                succState[succState < 0.0] += 1.0
                 
            return succState
        else:
            raise NotImplementedError()
    
    def getSuccessorDistribution(self, state, action):
        """ Returns an iterator that yields successor state probabilities
        
        Return an iterator that yields the pairs of state along with their 
        probabilities of being the successor state of *state* when *action*
        is performed by the agent.
        Note: This assummes a discrete (or discretized) state space since 
              otherwise this there will be infinitely many states with 
              probability > 0.      
        """
        if hasattr(self, "actionModels"):
            return self.actionModels[action].getSuccessorDistribution(state)
        else:
            raise NotImplementedError()
        
    def samplePredecessorState(self, state, action):
        """ Return sample predecessor state of state-action.
        
        Return a state drawn randomly from the predecessor distribution 
        of *state*-*action*
        """
        if hasattr(self, "actionModels"):
            return self.actionModels[action].samplePredecessorState(state)
        else:
            raise ModelNotInitialized()
    
    def getPredecessorDistribution(self, state, action):
        """ Returns an iterator that yields predecessor state probabilities
        
        Return an iterator that yields the pairs of state along with their 
        probabilities of being the predecessor state of *state* when *action*
        is performed by the agent.
        Note: This assumes a discrete (or discretized) state space since 
              otherwise this there will be infinitely many states with 
              probability > 0.      
        """
        if hasattr(self, "actionModels"):
            return self.actionModels[action].getPredecessorDistribution(state)
        else:
            raise NotImplementedError()
    
    def getExpectedReward(self, state, action):
        """ Returns expected reward when action is performed in state """
        if hasattr(self, "actionModels"):
            return self.actionModels[action].getExpectedReward(state)
        else:
            raise NotImplementedError()
    
    def getExplorationValue(self, state, action):
        """ Returns how often the pair state-action has been explored """
        if hasattr(self, "actionModels"):
            return self.actionModels[action].getExplorationValue(state)
        else:
            raise NotImplementedError()
        
    def getConfidence(self, state, action):
        """ 
        Return how confident the model is in its prediction for the given state
        action pair
        """
        if hasattr(self, "actionModels"):
            return self.actionModels[action].getConfidence(state)
        else:
            return 0.0

            
class JointStateActionModel(Model):
    """ Interface for models for continuous action spaces.
    
    The *JointStateActionModel* subclasses *Model* and changes its default 
    behaviour: Instead of forwarding every action to a separate ActionModel,
    state and action are concatenated and one ActionModel is used to learn
    the behaviour within this "State-Action-Space", i.e. a mapping from 
    (state, action_1) -> (succState, action_2). Since action_2 depends on the
    policy, it cannot be learned by a model and is thus ignored. For training
    of the ActionModel, action_2 is set to action_1.   
    
    NOTE: Currently, only one action dimension is supported!
    
    The constructor expects two parameter:
    * *actionRange* : A tuple indicating the minimal and maximal value of the
                      action space dimension.                         
    * *ActionModelClass* : A class implementing the ActionModel interface. This
                           class is used to learn the State-Action Space 
                           dynamics.
    """
    
    def __init__(self, agent, actionRange, actionSpace, ActionModelClass, **kwargs):
        # We only discretize the action space to allow for plotting
        # For prediction, we use the continuous action space 
        actions = numpy.linspace(actionRange[0], actionRange[1], 5)
        self.actionRange = actionRange
        self.actionSpace = actionSpace
        super(JointStateActionModel, self).__init__(agent,
                                                    actions = actions,
                                                    ActionModelClass = None,
                                                    **kwargs)
        
        self.model = ActionModelClass(**kwargs)
    
    def addExperience(self, state, action, succState, reward):
        """ Updates the model based on the given experience tuple
        
        Update the model based on the given experience tuple consisting
        of a *state*, an *action* taken in this state, the resulting
        successor state *succState* and the obtained *reward*.
        """
        # Scale action to range (0, 1)
        scaledAction = [(action[0] - self.actionRange[0]) \
                            / (self.actionRange[1] - self.actionRange[0])]
        super(JointStateActionModel, self).addExperience(state, scaledAction, 
                                                         succState, reward)
        
        stateAction = self._jointStateAction(state, action)
        succStateAction = self._jointStateAction(succState, action)
        self.model.addExperience(stateAction, succStateAction, reward)

    
    def getSample(self):
        """ Return a sample drawn randomly 
        
        Return a random sample (i.e. a state, action, reward, successor
        state 4-tuple).
        """
        try: 
            stateAction = self.model.sampleState()
            reward = self.getExpectedReward(stateAction)
            succState = self._extractState(self.sampleSuccessorState(stateAction))
            state = self._extractState(stateAction)
            action = stateAction[-1]
            # Rescale action from (0,1) to (actionRange[0], actionRange[1])
            action = action * (self.actionRange[1] - self.actionRange[0]) + self.actionRange[0]
            if succState == None:
                raise ModelNotInitialized("Not enough experience to sample from model")
            return (state, action, reward, succState)
        except IndexError:
            raise ModelNotInitialized("Not enough experience to sample from model")
            
    def sampleSuccessorState(self, state, action):
        """ Return sample successor state of state-action.
        
        Return a state drawn randomly from the successor distribution 
        of *state*-*action*
        """       
        stateAction = self._jointStateAction(state, action)
        succStateAction = self.model.sampleSuccessorState(stateAction)
        return self._extractState(succStateAction)
        
    def getSuccessorDistribution(self, state, action):
        """ Returns an iterator that yields successor state probabilities
        
        Return an iterator that yields the pairs of state along with their 
        probabilities of being the successor state of *state* when *action*
        is performed by the agent.
        Note: This assumes a discrete (or discretized) state space since 
              otherwise this there will be infinitely many states with 
              probability > 0.      
        """
        # TODO: deterministic
        return [(self.sampleSuccessorState(state, action), 1.0)]
        
    def samplePredecessorState(self, state, action):
        """ Return sample predecessor state of state-action.
        
        Return a state drawn randomly from the predecessor distribution 
        of *state*-*action*
        """
        stateAction = self._jointStateAction(state, action)
        predStateAction = self.model.samplePredecessorState(stateAction)
        return self._extractState(predStateAction)
    
    def getPredecessorDistribution(self, state, action):
        """ Returns an iterator that yields predecessor state probabilities
        
        Return an iterator that yields the pairs of state along with their 
        probabilities of being the predecessor state of *state* when *action*
        is performed by the agent.
        Note: This assumes a discrete (or discretized) state space since 
              otherwise this there will be infinitely many states with 
              probability > 0.      
        """
        raise NotImplementedError()
    
    def getExpectedReward(self, state, action):
        """ Returns expected reward when action is performed in state """
        stateAction = self._jointStateAction(state, action)
        return self.model.getExpectedReward(stateAction)
        
    
    def getExplorationValue(self, state, action):
        """ Returns how often the pair state-action has been explored """
        stateAction = self._jointStateAction(state, action)
        return self.model.getExplorationValue(stateAction)
        
    def getConfidence(self, state, action):
        """ 
        Return how confident the model is in its prediction for the given state
        action pair
        """
        stateAction = self._jointStateAction(state, action)
        return self.model.getConfidence(stateAction)
    
    def _jointStateAction(self, state, action):
        """ Create a joint state-action pseudo-state """                            
        dimensions = [dimension for dimension in state.dimensions] 
        actionDimension = copy.deepcopy(self.actionSpace.getDimensions()[0]) # there is per assert only 1
        dimensions.append(actionDimension)
        stateAction = State(numpy.hstack((state, action)), dimensions)
        stateAction.scale()
        return stateAction
    
    def _extractState(self, stateAction):
        """ Extracts the state from the joint state-action pseudo-state """
        dimensions = [dimension for dimension in stateAction.dimensions][:-1]
        state = State(stateAction[:-1], dimensions)
        return state


class InappropriateModelException(Exception):
    pass

        
class ActionModel(object):
    """ Interface for MMLF action models """
    
    def __init__(self, stateSpace, *args, **kwargs):
        if stateSpace.hasContinuousDimensions() \
                and "CONTINUOUS" not in self.__class__.SUPPORTED_STATE_SPACES:
            exceptionString = "Model %s does not support " \
                              "continuous state spaces." % self.__class__.__name__
            raise InappropriateModelException(exceptionString)
        elif not stateSpace.hasContinuousDimensions() \
                and "DISCRETE" not in self.__class__.SUPPORTED_STATE_SPACES:
            exceptionString = "Model %s does not support " \
                              "discrete state spaces." % self.__class__.__name__
            raise InappropriateModelException(exceptionString)
    
    def addExperience(self, state, succState, reward):
        """ Updates the action model based on the given experience tupel """
        raise NotImplementedError()
       
    def sampleState(self):
        """ Sample a state randomly from this action model """
        raise NotImplementedError() 
    
    def sampleSuccessorState(self, state):
        """ Sample a successor state for this state for this action-model """
        raise NotImplementedError()
    
    def getSuccessorDistribution(self, state):
        """ Iterates over pairs of successor states and their probabilities """ 
        raise NotImplementedError()

    def samplePredecessorState(self, state):
        """ Sample a predecessor state for this state for this action-model """
        raise NotImplementedError()
    
    def getPredecessorDistribution(self, state):
        """ Iterates over pairs of predecessor states and their probabilities """ 
        raise NotImplementedError()
    
    def getExpectedReward(self, state):
        """ Return the expected reward for the given state under this action"""
        raise NotImplementedError()
    
    def getExplorationValue(self, state):
        """ Returns the exploration value for the given state """
        raise NotImplementedError()
    
    def getConfidence(self, state):
        """ Return how confident the model is in its prediction for the given state """
        return 0.0
    
    def plot(self, ax, stateSpace, plotStateDims, dimValues, plotSamples, 
             colorFct, **kwargs):
        # Determine index of plot dimensions
        stateIndex1 = sorted(stateSpace.keys()).index(plotStateDims[0])
        stateIndex2 = sorted(stateSpace.keys()).index(plotStateDims[1])
        
        xValues = numpy.linspace(0, 1, dimValues[stateIndex1])
        yValues = numpy.linspace(0, 1, dimValues[stateIndex2])
        
        U = numpy.zeros((len(xValues), len(yValues)))
        V = numpy.zeros((len(xValues), len(yValues)))
        color = numpy.zeros((len(xValues), len(yValues)))
        for i in range(len(xValues)):
            for j in range(len(yValues)):
                numberOfDimensions = stateSpace.getNumberOfDimensions()
                node = numpy.zeros(numberOfDimensions)
                for k in range(numberOfDimensions):
                    if k == stateIndex1:
                        node[k] = xValues[i]
                    elif k == stateIndex2:
                        node[k] = yValues[j]
                    else:
                        node[k] = (dimValues[k]/2 + 0.5) / dimValues[k]
                
                node = State(node, 
                             [Dimension(sorted(stateSpace.keys())[dimNum],
                                        "continuous", 
                                        [[0,1]]) 
                                  for dimNum in range(numberOfDimensions)])

                # Find the maximum likely successor state
                p = 0.0
                maxSuccNode = node
                meanSuccNode = numpy.zeros(len(node))
                for succNode, prob in self.getSuccessorDistribution(node):
                    meanSuccNode += succNode*prob
                    if prob > p:
                        maxSuccNode = succNode
                        p = prob

                U[i,j] = meanSuccNode[stateIndex1] - node[stateIndex1]
                V[i,j] = meanSuccNode[stateIndex2] - node[stateIndex2]
                color[i,j] = colorFct(self, node, meanSuccNode)
    
        X,Y = numpy.meshgrid(xValues, yValues)
        ax.contourf(Y, X, color, 15)
#        pylab.colorbar()
        # Decide whether we plot the training samples or the predictions 
        if plotSamples:
            ax.scatter(self.states[:,stateIndex1], self.states[:,stateIndex2],
                          marker='o',c='b',s=5)
        else:
            ax.quiver(Y, X, U, V)
        ax.plot(range(0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(plotStateDims[0])
        ax.set_ylabel(plotStateDims[1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])                    
