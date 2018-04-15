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

""" Module for that implements various kinds of observables

Currently implemented observables are
 * FloatStreamObservable
 * TrajectoryObservable
 * StateActionValuesObservable
 * FunctionOverStateSpaceObservable
 * ModelObservable
"""

import warnings
import math
import numpy
from mmlf.gui.plot_utils import generate2dStateSlice, generate2dPlotArray

class Observable(object):
    """ Base class for all MMLf observables. """
    def __init__(self, title):       
        self.title = title
        self.observers = []
        # Add newly created observable
        OBSERVABLES.addObservable(self)
        
    def __del__(self):
        if self in OBSERVABLES.allObservables:
            OBSERVABLES.allObservables.removeObservable(self)
    
    def addObserver(self, observer):
        """ Add an observer that gets informed of all changes of this observable. """
        self.observers.append(observer)
        
    def removeObserver(self, observer):
        """ Remove an observer of this observable. """
        self.observers.remove(observer)
        
    def __repr__(self):
        return "Observable %s" % self.title
        
class AllObservables(Observable):
    """ The set of observables is itself observable, too """
    def __init__(self):
        self.title = "All Observables"
        self.observers = []
        
        self.allObservables = []
        self.addObservable(self)
        
    def addObservable(self, observable):
        """ Add a new observable to the set of all observables. """
        self.allObservables.append(observable)
        # Notify all observers by calling them
        for observer in self.observers:
            observer(observable, "added")

    def removeObservable(self, observable):
        """ Remove an observable from the set of all observables. """
        self.allObservables.remove(observable)
        # Notify all observers by calling them
        for observer in self.observers:
            observer(observable, "removed")
            
    def getAllObservablesOfType(self, type):
        """ Return all observables of a given type. """
        return filter(lambda ob: isinstance(ob, type), self.allObservables)
    
    def getObservable(self, observableName, type):
        """ Return all observable of a given type with given observableName."""
        for observable in self.getAllObservablesOfType(type):
            if observable.title == observableName:
                return observable

# A list that contains all existing observables
OBSERVABLES = AllObservables()
        
class FloatStreamObservable(Observable):
    """ An observable class that handles a stream of floats """
    def __init__(self, title, time_dimension_name, value_name):
        super(FloatStreamObservable, self).__init__(title)
        
        self.time_dimension_name = time_dimension_name
        self.value_name = value_name
        
    def addValue(self, time, value):
        """ Add a new value for the given point in time to the observable. """
        # Check if we have additional informations about agent and run number
        import threading
        ct = threading.currentThread()
        worldNumber = ct.worldNumber if hasattr(ct, "worldNumber") else None
        runNumber = ct.runNumber if hasattr(ct, "runNumber") else None
        
        # Notify all observers by calling them
        for observer in self.observers:
            observer(time, value, worldNumber, runNumber)
            

class TrajectoryObservable(Observable):
    """ An observable class that can monitor an agent's trajectory """
    def __init__(self, title):
        super(TrajectoryObservable, self).__init__(title)
                
    def addTransition(self, state, action, reward, succState, 
                      episodeTerminated=False):
        """ Add a new transition to the observable. 
        
        Adds the transition from *state* to *succState* for chosen action 
        *action* and obtained reward *reward* to the observable. 
        If *episodeTerminated*, succState is a terminal state of the environment.
        """
        # Notify all observers by calling them
        for observer in self.observers:
            observer(state, action, reward, succState, episodeTerminated)
            
      
class StateActionValuesObservable(Observable):
    """ Observable class for real valued functions over the state action space. 
    
    This might be for instance a Q-function or the eligibility traces or a 
    stochastic policy. 
    """
    def __init__(self, title):
        super(StateActionValuesObservable, self).__init__(title)
        
    def updateValues(self, valueAccessFunction, actions):
        """ Inform observable that the encapsulated function has changed."""
        # Notify all observers by calling them
        for observer in self.observers:
            observer(valueAccessFunction, actions)
            
    def plot(self, function, actions, fig, stateSpace, plotStateDims=None,
             plotActions=None, rasterPoints=100):
        """ Plots the q-Function for the case of a 2-dim subspace of the state space. 
        
        plotStateDims :The 2 dims that should be plotted
        plotActions : The action that should be plotted
        rasterPoints : How many raster points per dimension
        """ 
        # All actions that should be plotted
        if plotActions == None:
            plotActions = actions
        else:
            # Check if plot actions are valid actions
            for i in range(len(plotActions)):
                if plotActions[i] in actions: continue # ok...
                try:
                    plotActions[i] = eval(plotActions[i])
                except:
                    raise Exception("Invalid plot action %s" % plotActions[i])
                                               
        # Determine the indices of the dimension that should be plotted
        if plotStateDims == None or plotStateDims == []:
            if len(stateSpace.items()) != 2:
                warnings.warn("%s: Not two state space dimensions."
                              "Please specify plotStateDims explicitly. "
                                % self.__class__.__name__)
                return
            plotStateDims = [stateSpace.keys()[0], stateSpace.keys()[1]]
        elif len(plotStateDims) != 2:
            warnings.warn("%s: StateActionValuesObservable logging only defined when "
                          "2 plotStateDims  are explicitly specified."
                                % self.__class__.__name__)
            return
        
        # Prepare plotting
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                            wspace=0.1, hspace = 0.2)
        
        # Different plotting for discrete and continuous dimensions
        if stateSpace.hasContinuousDimensions():
            # Generate 2d state slice
            defaultDimValues = {}
            for dimensionName in stateSpace.keys():
                defaultDimValues[dimensionName] = 0.5
            stateSlice = generate2dStateSlice(plotStateDims, stateSpace,
                                              defaultDimValues,
                                              gridNodesPerDim=rasterPoints)       
    
            rows = int(math.ceil(len(plotActions)/2.0))
            data = dict()
            # determine absolute min and max to align subplots colormaps
            # initialized that will be exceeded in any case by the corresponding comparison
            absmin = float('inf')
            absmax = -float('inf')
            # For all actions that should be plotted
            for plotNum, action in enumerate(plotActions):            
                # Compute values that should be plotted, colormapping == {} in continuous case
                values, colorMapping = \
                        generate2dPlotArray(lambda state : function(state, action),
                                            stateSlice, True, 
                                            shape=(rasterPoints, rasterPoints)) 
                
                # Check if there is something to plot
                if values.mask.all():
                    continue

                # some comparison for colormap alignment
                thismin = values.min()
                if thismin < absmin:
                    absmin = thismin
                thismax = values.max()
                if thismax > absmax:
                    absmax = thismax

                # save the data to plot later, when ranges are known
                data[(plotNum,action)] = values.T

            # plot the data
            for plotNum, action in data.keys():
                # add subplot
                subplot = fig.add_subplot(rows, 2, plotNum + 1)
                subplot.clear()

                # create pseudocolorplot in current subplot
                polyCollection = fig.gca().pcolor(
                    numpy.linspace(0.0, 1.0, rasterPoints),
                    numpy.linspace(0.0, 1.0, rasterPoints),
                    data[(plotNum,action)], vmin = absmin, vmax = absmax)
                
                # Add colorbar
                fig.colorbar(polyCollection)
    
                # Labeling etc.
                subplot.set_xlim(0,1)
                subplot.set_ylim(0,1)
                subplot.set_xlabel(plotStateDims[0])
                subplot.set_ylabel(plotStateDims[1])
                subplot.set_title(action)
        else:
            assert (len(stateSpace.items()) == 2), \
                    "Discrete state spaces can only be plotted if they have two dimensions."
            valuesX = stateSpace[plotStateDims[0]]["dimensionValues"]
            valuesY = stateSpace[plotStateDims[1]]["dimensionValues"]
            stateSlice = {}
            from mmlf.framework.state import State
            for i, valueX in enumerate(valuesX):
                for j, valueY in enumerate(valuesY):
                    # Create state object
                    stateSlice[(i,j)] = State([valueX, valueY], 
                                              [stateSpace[plotStateDims[0]], 
                                               stateSpace[plotStateDims[1]]])
                    
            rows = int(math.ceil(len(plotActions)/2.0))
            # For all actions that should be plotted
            for plotNum, action in enumerate(plotActions):            
                # Clear old plot
                subplot = fig.add_subplot(rows, 2, plotNum + 1)
                subplot.clear()
                # Compute values that should be plotted
                values, colorMapping = \
                        generate2dPlotArray(lambda state : function(state, action),
                                            stateSlice, True, 
                                            shape=(len(valuesX), len(valuesY))) 
                
                # Check if there is something to plot
                if values.mask.all():
                    continue
                
                # Do the actual plotting
                polyCollection = \
                    fig.gca().pcolor(numpy.array(valuesX) - 0.5, 
                                     numpy.array(valuesY) - 0.5, values.T)
                
                # Add colorbar
                fig.colorbar(polyCollection)
    
                # Labeling etc.
                subplot.set_xlim(min(valuesX), max(valuesX))
                subplot.set_ylim(min(valuesY), max(valuesY))
                subplot.set_xlabel(plotStateDims[0])
                subplot.set_ylabel(plotStateDims[1])
                subplot.set_title(action)           
            
      
class FunctionOverStateSpaceObservable(Observable):
    """ Observable class for observing functions over the state space.
    
    This class implements an observable for observing (one-dimensional) functions
    defined over the state space. This function can either implement an mapping
    f: S -> R (real numbers, *discreteValues*=False) or a mapping f: S -> C 
    (some discrete finite set of values, *discreteValues*=True). An example 
    for the first kind of function would be the optimal value function V(s), an
    example for the second case a deterministic policy pi(s), with C being the
    action space.
    """
    def __init__(self, title, discreteValues):
        super(FunctionOverStateSpaceObservable, self).__init__(title)
        self.discreteValues = discreteValues
        
        
    def updateFunction(self, function):
        """ Inform observable that the encapsulated function has changed."""
        # Notify all observers by calling them
        for observer in self.observers:
            observer(function)
            
    def plot(self, function, fig, stateSpace, actionSpace, 
             plotStateDims=None, rasterPoints=100):
        """ Creates a graphical representation of a FunctionOverStateSpace.
        
        Creates a plot of *policy* in the 2D subspace of the state space
        spanned by *stateIndex1* and *stateIndex2*. 
        """        
        # Determine the indices of the dimension that should be plotted
        if plotStateDims == None or plotStateDims == []:
            if len(stateSpace.items()) != 2:
                warnings.warn("%s: Not two state space dimensions. "
                              "Please specify plotStateDims explicitly. "
                                % self.__class__.__name__)
                return
            plotStateDims = [stateSpace.keys()[0], stateSpace.keys()[1]]
        elif len(plotStateDims) != 2:
            warnings.warn("%s: FunctionOverStateSpace logging only defined when "
                          "2 plotStateDims  are explicitly specified."
                                % self.__class__.__name__)
            return
                     
        # Prepare plotting
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                            wspace=0.1, hspace = 0.1)
        
        # Different plotting for discrete and continuous dimensions
        if stateSpace.hasContinuousDimensions():
            # Generate 2d state slice
            defaultDimValues = {}
            for dimensionName in stateSpace.keys():
                defaultDimValues[dimensionName] = 0.5
            stateSlice = generate2dStateSlice(plotStateDims, stateSpace,
                                              defaultDimValues,
                                              gridNodesPerDim=rasterPoints)
        
            # Compute values that should be plotted
            values, colorMapping = \
                        generate2dPlotArray(function, stateSlice, 
                                            not self.discreteValues,
                                            shape=(rasterPoints, rasterPoints))
            
            # Check if there is something to plot
            if values.mask.all():
                return 
            
            # Do the actual plotting
            polyCollection = \
                fig.gca().pcolor(numpy.linspace(0.0, 1.0, rasterPoints),
                                 numpy.linspace(0.0, 1.0, rasterPoints),
                                 values.T)
            
            # Polishing of figure
            fig.gca().set_xlim(0.0, 1.0)
            fig.gca().set_ylim(0.0, 1.0)
        else:
            assert (len(stateSpace.items()) == 2), \
                    "Discrete state spaces can only be plotted if they have two dimensions."
            valuesX = stateSpace[plotStateDims[0]]["dimensionValues"]
            valuesY = stateSpace[plotStateDims[1]]["dimensionValues"]
            stateSlice = {}
            from mmlf.framework.state import State
            for i, valueX in enumerate(valuesX):
                for j, valueY in enumerate(valuesY):
                    # Create state object
                    stateSlice[(i,j)] = State([valueX, valueY], 
                                              [stateSpace[plotStateDims[0]], 
                                               stateSpace[plotStateDims[1]]])
                    
            # Compute values that should be plotted
            values, colorMapping = \
                        generate2dPlotArray(function, stateSlice, 
                                            not self.discreteValues,
                                            shape=(len(valuesX), len(valuesY)))
                    
            polyCollection = \
                fig.gca().pcolor(numpy.array(valuesX) - 0.5, 
                                 numpy.array(valuesY) - 0.5,
                                 values.T)
        
            # Polishing of figure
            fig.gca().set_xlim(min(valuesX), max(valuesX))
            fig.gca().set_ylim(min(valuesY), max(valuesY))
            
        fig.gca().set_xlabel(plotStateDims[0])
        fig.gca().set_ylabel(plotStateDims[1])            
        # Create legend respective colorbar
        if not self.discreteValues:
            fig.colorbar(polyCollection)
        else:
            # Some dummy code that creates patches that are not shown but allow
            # for a colorbar
            from matplotlib.patches import Rectangle
            linearSegmentedColorbar = polyCollection.get_cmap()
            patches = []
            functionValues = []
            for functionValue, colorValue in colorMapping.items():
                if isinstance(functionValue, tuple):
                    functionValue = functionValue[0] # deal with '(action,)'
                normValue = polyCollection.norm(colorValue)
                if isinstance(normValue, numpy.ndarray): 
                    normValue = normValue[0] # happens when function is constant
                rgbaColor = linearSegmentedColorbar(normValue)
             
                p = Rectangle((0, 0), 1, 1, fc=rgbaColor)
                functionValues.append(functionValue)
                patches.append(p)
            fig.gca().legend(patches, functionValues)


class ModelObservable(Observable):
    """ Observable class for observing models.
    
    This class implements an observable for observing MMLF models. Whenever a 
    new state transition, a new start state or a new terminal state is added to
    the model, this observable notifies all observers of this change.
    """
    def __init__(self, title):
        super(ModelObservable, self).__init__(title)
        
    def updateModel(self, model):
        """ Inform observable that the encapsulated model has changed."""
        # Notify all observers by calling them
        for observer in self.observers:
            observer(model)
            
    def plot(self, model, fig, stateSpace, colouring, plotSamples, 
             minExplorationValue, plotStateDims, dimValues):
        """ Does the actual plotting (either for viewer or for log-graphic). """
        from mmlf.resources.model.model import ModelNotInitialized
        
        # Determine the indices of the dimension that should be plotted
        if plotStateDims == None or plotStateDims == []:
            if len(stateSpace.items()) != 2:
                warnings.warn("%s: Not two state space dimensions."
                              "Please specify plotStateDims explicitly. "
                                % self.__class__.__name__)
                return
            plotStateDims = [stateSpace.keys()[0], stateSpace.keys()[1]]
        elif len(plotStateDims) != 2:
            warnings.warn("%s: Model logging only defined when 2 plotStateDims "
                          "are explicitly specified."
                                % self.__class__.__name__)
            return
        
        # Determine colouring
        if colouring == "Exploration":
            def colorFct(actionModel, state, succState):
                return 0.0 if actionModel.getExplorationValue(state) < minExplorationValue else 1.0
        elif colouring == "Rewards":
            def colorFct(actionModel, state, succState):
                return numpy.nan if model.isTerminalState(state) else actionModel.getExpectedReward(state)
        else:
            raise Exception("Invalid colouring for model plotting is specified.")
        
        if plotStateDims is None:
            if stateSpace.getNumberOfDimensions() != 2:
                raise Exception("Dimensions that should be plotted must be "
                                "explicitly defined if state space is not 2d.")
            else:
                plotStateDims = sorted(stateSpace.keys())
        else:
            plotStateDims = sorted(plotStateDims)
        
        # Define dim values
        if not hasattr(dimValues, "__iter__"):
            dimValues = [dimValues] * stateSpace.getNumberOfDimensions()
            
        # Do the actual plotting
        fig.clear()        
        fig.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.93,
                            wspace=0.1, hspace=0.1)

        rows = int(math.ceil(len(model.actions)/2.0))
        try:
            for plotNumber, action in enumerate(model.actions):
                ax = fig.add_subplot(rows, 2, plotNumber + 1)
                model.actionModels[action].plot(ax, stateSpace=stateSpace, 
                                                plotStateDims=plotStateDims,
                                                dimValues=dimValues,
                                                plotSamples=plotSamples,
                                                colorFct=colorFct, 
                                                isStartState= lambda state: state in model.startStates,
                                                isTerminalState=lambda state: model.isTerminalState(state))               
                ax.set_title("Action: %s" % action, fontsize=14)
        except ModelNotInitialized, e:
            pass   

