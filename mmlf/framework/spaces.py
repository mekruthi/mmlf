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

""" Modules for state and action spaces

State and action spaces define the range of possible states the agent might
perceive and the actions that are available to the agent. These spaces are 
dict-like objects that map dimension names (the dict keys) to dimension objects 
that contain information about this dimension. The number of items in this 
dict-like structure is the dimensionality of the (state/action) space.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edginton']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random

from mmlf.framework.state import State

class Dimension(dict):
    """ A single dimension of a (state or action) space
    
    A dimension is either continuous or discrete. A "discrete" dimension
    might take on only a finite, discrete number of values. 
    
    For instance, consider a dimension of a state space describing the color of
    a fruit. This dimension take on the values "red", "blue", or "green". 
    In contrast, consider a second "continuous" dimension, e.g. the weight of 
    a fruit. This weight might be somewhere between 0g and 1000g. 
    If we allow any arbitrary weight (not only full gramms), the dimension is 
    truly continuous. 
    
    This properties of a dimension can be checked using the method:
     * *isDiscrete* : Returns whether the respective dimension is discrete
     * *isContinuous* : Returns whether the respective dimension is continuous
     * *getValueRanges* : Returns the allowed values a continuous! dimension
                          might take on.
     * *getValues* : Returns the allowed values a discrete! dimension
                     might take on. 
    """
    def __init__(self, dimensionName, dimensionType, dimensionValues, 
                 limitType = None):
        super(Dimension, self).__init__()
        assert(dimensionType == "continuous" or limitType == None)
        
        self["dimensionName"] = dimensionName
        self["dimensionType"] = dimensionType
        self["dimensionValues"] = dimensionValues
        self["limitType"] = limitType
        
    def __repr__(self):
        if self["dimensionType"] == "continuous":
            s = "ContinuousDimension(LimitType: %s, Limits: (%.3f,%.3f))" \
                    % (self["limitType"], self["dimensionValues"][0][0], 
                       self["dimensionValues"][0][1])
        else:
            s = "DiscreteDimension(Values: %s)" % self["dimensionValues"]
            
        return s  
        
    def isContinuous(self):
        """ Returns whether this is a continuous dimension """
        return self["dimensionType"] == "continuous"
        
    def isDiscrete(self):
        """ Returns whether this is a discrete dimension """
        return self["dimensionType"] == "discrete"
    
    def getValueRanges(self):
        """ Returns the ranges the value of this dimension can take on """
        assert(self.isContinuous())
        return self["dimensionValues"]
    
    def getValues(self):
        """ Returns a list of all possible values of this dimension """
        assert(self.isDiscrete())
        return self["dimensionValues"]

class Space(dict):
    """ Base class for state and action spaces.
    
    Class which represents the state space of an environment or the action space
    of an agent.
    
    This is essentially a dictionary whose keys are the names of the dimensions,
    and whose values are *Dimension* objects.    
    """
    def __init__(self):
        pass
    
    def __repr__(self):
        s = "\n\t%s:" % self.__class__.__name__
        for dimensionName, dimensionDefinition in self.iteritems():
            s += "\n\t\t%-15s : %s" % (dimensionName, dimensionDefinition)
        return s
    
    def addContinuousDimension(self, dimensionName, dimensionValues, limitType="soft"):
        """
        Add the named continuous dimension to the space.
        
        dimensionValues is a list of (rangeStart, rangeEnd) 2-tuples which define the valid ranges
        of this dimension. (i.e. [(0, 50), (75.5, 82)] )
        
        If limitType is set to "hard", then the agent is responsible to check that the limits are
        not exceeded.  When it is set to "soft", then the agent should not expect that all the values
        of this dimension will be strictly within the bounds of the specified ranges, but that
        the ranges serve as an approximate values of where the values will be (i.e. as
        [mean-std.dev., mean+std.dev] instead of [absolute min. value, absolute max. value])
        
        """
        assert limitType in ("soft", "hard")
        
        self[dimensionName] = Dimension(dimensionName = dimensionName,
                                        dimensionType = "continuous",
                                        dimensionValues = dimensionValues,
                                        limitType = limitType)
    
    def addDiscreteDimension(self, dimensionName, dimensionValues):
        """
        Add the named continuous dimension to the space.
        
        dimensionValues is a list of strings representing possible discrete states of this dimension.
        (i.e. ["red", "green", "blue"])
        """        
        self[dimensionName] = Dimension(dimensionName = dimensionName,
                                        dimensionType = "discrete",
                                        dimensionValues = dimensionValues)
    
    def addOldStyleSpace(self, oldStyleSpace, limitType="soft"):
        """
        Takes an old-style (using the old format) space dictionary, and adds its dimensions to this
        object.
        """
        for dimensionName, dimensionDescriptionList in oldStyleSpace.iteritems():
            dimensionType = dimensionDescriptionList[0]
            dimensionValues = dimensionDescriptionList[1]
            if dimensionType == "discrete":
                self.addDiscreteDimension(dimensionName, dimensionValues)
            elif dimensionType == "continuous":
                self.addContinuousDimension(dimensionName, dimensionValues, limitType=limitType)
    
    def getDimensions(self):
        """ Return the names of the space dimensions """
        return [value for key, value in sorted(self.items())]
    
    def getDimensionNames(self):
        """ Return the names of the space dimensions """
        return self.keys()
    
    def getNumberOfDimensions(self):
        """ Returns how many dimensions this space has """
        return len(self.keys())
    
    def hasContinuousDimensions(self):
        """ Return whether this space has continuous dimensions """
        for dimension in self.itervalues():
            if dimension.isContinuous():
                return True
        return False
    
    def hasDiscreteDimensions(self):
        """ Return whether this space has discrete dimensions """
        for dimension in self.itervalues():
            if dimension.isDiscrete():
                return True
        return False
        
        
class StateSpace(Space):
    """ Specialization of Space for state spaces.
    
    For instance, a state space could be defined as follows::

        { "color": Dimension(dimensionType = "discrete",
                             dimensionValues = ["red","green", "blue"]),
          "weight": Dimension(dimensionType = "continuous",
                              dimensionValues = [(0,1000)]) }
                              
    This state space has two dimensions ("color" and "weight"), a discrete and a 
    continuous one. The discrete dimension "color" can take on three values
    ("red","green", or "blue") and the continuous dimension "weight" any value
    between 0 and 1000.
    
    A valid state of the state space defined above would be::
    
        s1 = {"color": "red", "weight": 300}

    Invalid states (s2 since the color is invalid and s3 since its weight is
    too large)::
        s2 = {"color": "yellow", "weight": 300}
        s3 = {"color": "red", "weight": 1300}

    The class provides additional methods for checking if a certain state is  
    valid according to this state space (*isValidState*) and to scale a state 
    such that it lies within a certain interval (*scaleState*).    
    """
    
    def parseStateDict(self, stateDict):
        #Check whether the given state dict is a valid one
        assert self._isValidState(stateDict), "State %s is invalid!" % stateDict
        
        state = State([stateDict[key] for key in sorted(stateDict.keys())], 
                      map(lambda name: self[name], sorted(stateDict.keys())))
        
        return state
    
    def getStateList(self):
        """ Returns a list of all possible states.
        
        Even if this state space has more than one dimension, it returns a one
        dimensional list that contains all possible states.        
        
        This is achieved by creating the crossproduct of the values of
        all dimensions. It requires that all dimensions are discrete.
        """
        # Check that all dimensions are discrete
        for dimension in self.getDimensions():
            assert dimension.isDiscrete(), \
                   "State list are available only for discrete state spaces!"
        
        #Create cross product of all possible actions
        crossProduct = lambda ss, row=[], level=0: \
                            len(ss)>1 \
                            and reduce(lambda x,y:x+y,[crossProduct(ss[1:],row+[i],level+1) for i in ss[0]]) \
                            or [row+[i] for i in ss[0]]
        
        listOfStateDimensionValues = [dimension.getValues()
                                           for dimension in self.getDimensions()]
        
        # Return crossproduct of states
        return map(lambda value: State(value, self.getDimensions()),
                   crossProduct(listOfStateDimensionValues))
    
    def _isValidState(self, state):
        """ Checks if the given state is valid according to this state space """
        for name, value in state.iteritems():
            if name not in self.getDimensionNames():
                return False
            
            dimension = self[name]
            
            if dimension.isDiscrete():
                return (value in dimension.getValues())
            else:
                if dimension["limitType"] == "soft":
                    return True
                else :
                    for valueRange in dimension.getValueRanges():
                        if valueRange[0] <= value <= valueRange[1]:
                            return True
            
            return False
        


class ActionSpace(Space):
    """ Specialization of Space for action spaces.
    
    For instance, an action space could be defined as follows::

      { "gasPedalForce": ("discrete", ["low", "medium", "floored"]),
        "steeringWheelAngle": ("continuous", [(-120,120)]) }
                              
    This action space has two dimensions ("gasPedalForce" and
    "steeringWheelAngle"), a discrete and a continuous one. 
    The discrete dimension "gasPedalForce" can take on three values
    ("low","medium", or "floored") and the continuous dimension 
    "steeringWheelAngle" any value between -120 and 120.
    
    A valid action according to this action space would be::
    
        a1 = {"gasPedalForce": "low", "steeringWheelAngle": -50}

    Invalid actions (a2 since the gasPedalForce is invalid and s3 since its 
    steeringWheelAngle is too small)::
        a2 = {"gasPedalForce": "extreme", "steeringWheelAngle": 30}
        a3 = {"gasPedalForce": "medium", "steeringWheelAngle": -150}

    The class provides additional methods for discretizing an action space
    (*discretizedActionSpace*) and to return a list of all available actions
    (*getActionList*).
    """   
    
    def discretizedActionSpace(self, discreteActionsPerDimension):
        """ Return a discretized version of this action space 
        
        Returns a discretized version of this action space. Every continuous
        action space dimension is discretized into *discreteActionsPerDimension*
        actions.
        """
        discretizedActionSpace = ActionSpace()
        for name, dimension in self.iteritems():
            if dimension.isContinuous(): #we have to discretize it
                #Get the minimal and maximal value of the continuous range
                assert len(dimension.getValueRanges()) == 1, \
                    "Discretizing action dimension requires that all dimensions "\
                    "have contiguous value ranges."
                minValue, maxValue = dimension.getValueRanges()[0] ##TODO
                #replace the continuous def by a discrete one
                discreteActions = [minValue + float(i)/(2*discreteActionsPerDimension)*(maxValue - minValue) \
                                    for i in range(1, 2*discreteActionsPerDimension+1, 2)]
                discretizedActionSpace[name] = Dimension(dimensionName = name,
                                                         dimensionType = "discrete",
                                                         dimensionValues = discreteActions)
            else:
                discretizedActionSpace[name] = dimension
                
        return discretizedActionSpace
    
    def getActionList(self):
        """ Returns a list of all allowed action an agent might take. 
        
        Even if this action space has more than one dimension, it returns a one
        dimensional list that contains all allowed action combinations.        
        
        This is achieved by creating the crossproduct of the values of
        all dimensions. It requires that all dimensions are discrete.    
        """
        # Check that all dimensions are discrete
        for dimension in self.getDimensions():
            assert dimension.isDiscrete(), \
                   "Action list are available only for discrete action spaces!"
        
        #Create cross product of all possible actions
        crossProduct = lambda ss, row=[], level=0: \
                            len(ss)>1 \
                            and reduce(lambda x,y:x+y,[crossProduct(ss[1:],row+[i],level+1) for i in ss[0]]) \
                            or [row+[i] for i in ss[0]]
        
        listOfActionDimensionValues = [dimension.getValues()
                                           for dimension in self.getDimensions()]
        
        # Return crossproduct of actions
        return map(tuple, crossProduct(listOfActionDimensionValues))
        
    
    def sampleRandomAction(self):
        """ Returns a randomly sampled, valid action from the action space. """
        actionDictionary = {} # initialize dictionary for storing action to be taken
        for keyName in self.keys():
            actionDimensionType = self[keyName]["dimensionType"]
            actionValueList = self[keyName]["dimensionValues"]
            
            if actionDimensionType == "discrete":
                # choose one of the available discrete actions
                actionIndex = random.randint(0, len(actionValueList) - 1)
                action = actionValueList[actionIndex] # a string
            
            if actionDimensionType == "continuous":
                # choose one of the available range-tuples
                rangeTupleIndex = random.randint(0, len(actionValueList) - 1)
                rangeTuple = actionValueList[rangeTupleIndex]
                # choose a value within the range of this tuple
                action = random.uniform(rangeTuple[0], rangeTuple[1]) # a floating-point number
                
            actionDictionary[keyName] = action
            
        return actionDictionary
    
    def chopContinuousAction(self, action):
        """ Chop a continuous action into the range of allowed values. """
        assert self.getNumberOfDimensions() == 1, \
                "Chopping actions works currently only for one-dimensional "\
                "action spaces"
                
        dimension = self.getDimensions()[0]
        if action <  dimension["dimensionValues"][0][0]:
            action = dimension["dimensionValues"][0][0]
        elif action >  dimension["dimensionValues"][0][1]:
            action = dimension["dimensionValues"][0][1]
        return action
    