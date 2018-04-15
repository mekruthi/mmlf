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
# Created: 2009/07/20
""" An action model class based on LWPR state transition modeling. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


import random
import numpy

if "sphinx" not in __import__("sys").argv[0]: # building documentation
    import lwpr
    import scikits.ann as ann

from mmlf.framework.state import State
from mmlf.resources.model.model import ActionModel, ModelNotInitialized
from mmlf.resources.model.example_set import ExampleSet

class LWPRModel(ActionModel):
    """ An action model class based on LWPR state transition modeling.
    
    This model learns the state successor (and predecessor) function using the 
    "Locally Weighted Projection Regression" (LWPR) regression learner. This
    learner learns a deterministic model, i.e. each state is mapped onto the
    successor state the learner considers to be most likely (and thus not onto
    a probability distribution).
    
    The reward function is learned using a nearest neighbor (NN) model. The
    reason that NN is used and not LWPR is that the reward function is usually
    non-smooth and every more sophisticated learning scheme might introduce
    additional unjustified bias.
    
    This model stores a fixed number of example transitions in a so called 
    example set and relearns the model whenever necessary (i.e. when predictions
    are requested and new examples have been addedsince the last learning).
    The model is relearned from scratch. 
    
    **CONFIG DICT**
        :exampleSetSize: : The maximum number of example transitions that is remembered in the example set. If the example set is full, old examples must be deleted or now new examples are accepted.
        :examplesPerModelUpdate: : The number of examples presented to LWPR before the learning is stopped
        :init_d: : The init_d parameter for LWPR that controls the smoothness of the learned function. Smaller values correspond to more smooth functions.   
    """
    
    DEFAULT_CONFIG_DICT = {'exampleSetSize':  1000,
                           'examplesPerModelUpdate' : 10000,
                           'init_d' : 25}
    
    SUPPORTED_STATE_SPACES = ["CONTINUOUS"]
    
    def __init__(self, stateSpace, exampleSetSize=1000, 
                 examplesPerModelUpdate=10000, init_d=25,  **kwargs):
        super(LWPRModel, self).__init__(stateSpace, **kwargs)
        
        # How many examples are presented to the model per update
        self.examplesPerModelUpdate = examplesPerModelUpdate
        
        # A set of state - reward - succ state transitions
        self.exampleSet = ExampleSet(exampleSetSize)
        
        # Scaling of the distance matrix of LWPR, controls the smoothness
        self.init_d = float(init_d)
        
        # The dimensionality of the state space
        self.stateDims = None # Is set lazily 
        
        # successor state distribution
        self.succStateModel = None 
        # predecessor state distribution
        self.predStateModel = None 
        # reward model
        self.rewardModel = dict()
        
        # Store whether samples have been added since the last update
        self.retrainingRequired = False     
    
    def addExperience(self, state, succState, reward):
        """ Add the given experience tuple to the example set of the model.
        
        Updates the model with one particular experience triple (s, s',r).
        The action need not be specified since this model is for one fixed 
        action only.If the example set is full, the given example is either 
        rejected or an old experience tuple is removed. Such a tuple is removed
        if there are examples in the set that are more similar to each other 
        than the given example to its nearest neighbor in the set.
        """       
        # Lazy initialization  (dimensionality only known after first sample
        # received):
        if self.succStateModel == None:
            self.stateDims = state.shape[0]
    
        # If state is already known
        if state in self.rewardModel:
            return
        
        # Add the transition to the example set, potentially replacing
        # an older transition if the maximum number of transitions has been
        # reached
        removedState = self.exampleSet.addTransition(state, succState, reward)
        
        # Delete the reward information of the old example and add the new one
        if removedState is not None:
            self.rewardModel.pop(removedState)
        self.rewardModel[state] = reward
        
        self.retrainingRequired = True
        
    def sampleState(self):
        """ Return a state drawn randomly """
        stateDensity = self.exampleSet.getStateDensity()
        if stateDensity != None:
            # TODO: Does it make sense to sample based on the data set? 
            return State(stateDensity.resample(1).T[0])
        else:
            raise ModelNotInitialized()
    
    def sampleSuccessorState(self, state):
        """ Return a state drawn from the state's successor distribution """
        if self._retrainingRequired():
            self._updateModel()
        if self.succStateModel != None:
            return State(state + self.succStateModel.predict(state),
                         state.dimensions)
        else:
            raise ModelNotInitialized()
        
    def getSuccessorDistribution(self, state):
        """ Return the successor distribution for the given state. 
        
        Returns an iterator that yields pairs of grid nodes and
        their probabilities of being the successor of the given state. 
        """
        if self._retrainingRequired():
            self._updateModel()
            
        if self.succStateModel != None:
            # This is a deterministic model!
            yield (State(state + self.succStateModel.predict(state),
                         state.dimensions),
                   1.0)
        else:
            raise ModelNotInitialized()
    
    def samplePredecessorState(self, state):
        """ Return a state drawn from the state's predecessor distribution """
        if self._retrainingRequired():
            self._updateModel()
            
        if self.predStateModel != None:
            return State(state + self.predStateModel.predict(state),
                         state.dimensions)
        else:
            raise ModelNotInitialized()
    
    def getPredecessorDistribution(self, state):
        """ Return the predecessor distribution for the given state. 
        
        Returns an iterator that yields pairs of states and
        their probabilities of being the predecessor of the given state. 
        """
        if self._retrainingRequired():
            self._updateModel()
            
        if self.predStateModel != None:
            state = numpy.array(state)
            # This is a deterministic model!
            yield (State(state + self.predStateModel.predict(state),
                         state.dimensions),
                   1.0)
        else:
            raise ModelNotInitialized()
    
    def getExpectedReward(self, state):
        """ Returns the expected reward for the given state *state* """
        if self._retrainingRequired():
            self._updateModel()
            
        if self.rewardModel != None and self.exampleSet.states != None:
            nearestNeighbor = self.exampleSet.getNearestNeighbor(state)
            return float(self.rewardModel[nearestNeighbor])
        else:
            raise ModelNotInitialized()
    
    def getExplorationValue(self, state):
        """ Return the exploratory value of the given state *state*
        
        The exploratory value of a state under this model is defined simply as
        the euclidean distance of from the state to its nearest neighbor in the
        example set. 
        """
        if self._retrainingRequired():
            self._updateModel()
                   
        if self.exampleSet.states != None:
            nearestNeighbor = self.exampleSet.getNearestNeighbor(state)
            dist = numpy.linalg.norm(nearestNeighbor - state)
            return -dist
        else:
            raise ModelNotInitialized()
    
    def getConfidence(self, state):
        """ 
        Return how confident the model is in its prediction for the given state
        """
        if self.succStateModel != None:
            return max(self.succStateModel.predict_conf(state)[1])
        else:
            return 0.0
        
    def _retrainingRequired(self):
        """ Return whether the model needs to be retrained 
        
        The model needs to be retrained when new examples are available.
        """
        return self.retrainingRequired
        
    def _updateModel(self):
        """ Updates the internal models based on the gathered examples """
        self.succStateModel = lwpr.LWPR(self.stateDims, self.stateDims)
        self.predStateModel = lwpr.LWPR(self.stateDims, self.stateDims)
        self.succStateModel.init_D = \
            numpy.diag([self.init_d 
                            for i in range(self.succStateModel.init_D.shape[1])])
        self.predStateModel.init_D = \
            numpy.diag([self.init_d
                            for i in range(self.predStateModel.init_D.shape[1])])
        
        # Update models with randomly drawn examples
        for transition in self.exampleSet.drawTransitions(self.examplesPerModelUpdate):
            state, succState, reward = transition         
            self.succStateModel.update(state, succState - state)
            self.predStateModel.update(succState, state - succState)
        
        self.retrainingRequired = False
            
    def plotSuccStateError(self, stateSpace, logFile):
        """
        Estimate the error the model makes and plot this to the given logFile
        """
        if self.stateDims is None or self.exampleSet is None:
            return
        
        numberOfSamples = self.exampleSet.states.shape[1]
        if numberOfSamples <= 3:
            return
                
        succStateModel = lwpr.LWPR(self.stateDims, self.stateDims)
        succStateModel.init_D = numpy.diag([self.init_d for i in range(succStateModel.init_D.shape[1])])        
        # Split data into train and test set
        trainSet = []
        testSet = []
        for counter, index in enumerate(numpy.random.permutation(range(numberOfSamples))):
            sample = (self.exampleSet.states[:, index],
                      self.exampleSet.succStates[:, index],
                      self.exampleSet.rewards[:, index])
            if counter < numberOfSamples * 0.66:
                trainSet.append(sample)
            else:
                testSet.append(sample)
        
        # Train the model
        for i in range(10000):
            transition = random.choice(trainSet)
            state, succState, reward = transition         
            succStateModel.update(state, succState - state)

        # Determine error of test set samples
        errors = {}
        for transition in testSet:
            state, succState, reward = transition
            state = numpy.array(list(state))
            succState = numpy.array(list(succState))
            errors[tuple(transition[0])] = numpy.linalg.norm(state + succStateModel.predict(state) - succState)
        
        # Plot error
        import pylab
        import matplotlib
        
        pylab.figure(100)
        contour = pylab.contourf(zip(errors.values(),errors.values()))
        
        fig = pylab.figure(0, figsize = (24,13))
        pylab.subplots_adjust(left  = 0.05,  # the left side of the subplots of the figure
                              right = 0.95,    # the right side of the subplots of the figure
                              bottom = 0.05,   # the bottom of the subplots of the figure
                              top = 0.95,      # the top of the subplots of the figure
                              wspace = 0.1,   # the amount of width reserved for blank space between subplots
                              hspace = 0.1,   # the amount of height reserved for white space between subplots
                              )
        fig.clear()
        
        normalizer = matplotlib.colors.Normalize(vmin=numpy.min(errors.values()),
                                                 vmax=numpy.max(errors.values()))
        
        for dim1 in range(self.stateDims):
            for dim2 in range(dim1 + 1, self.stateDims):
                pylab.subplot(self.stateDims - 1,  self.stateDims,
                              (self.stateDims - 1) * dim1 + dim2 + 1)
                dim1Values = []
                dim2Values = []
                errorValues = []
                for state, error in errors.iteritems():
                    dim1Values.append(state[dim1])
                    dim2Values.append(state[dim2])
                    errorValues.append(error)
                
                # define grid.
                xi = numpy.linspace(0.0, 1.0, 100)
                yi = numpy.linspace(0.0, 1.0, 100)
                # grid the data.
                zi = pylab.griddata(numpy.array(dim2Values), 
                                    numpy.array(dim1Values),
                                    numpy.array(errorValues), xi, yi)
                # contour the gridded data
                CS = pylab.contour(xi, yi, zi, 15, linewidths=0.5, colors='k',
                                   norm=normalizer)
                CS = pylab.contourf(xi, yi, zi, 15, norm=normalizer)

                pylab.xlim(0, 1)
                pylab.ylim(0, 1)
                pylab.xlabel(stateSpace.keys()[dim2])
                pylab.ylabel(stateSpace.keys()[dim1])

        ax = pylab.subplot(self.stateDims - 1,  self.stateDims,
                        (self.stateDims - 1) * (self.stateDims - 2) + 1)
        pylab.colorbar(mappable=contour, cax=ax)
        
        # Save the plot to the log file
        fig.savefig(logFile)        
        fig.clear()        
        
                                          
                                          
        