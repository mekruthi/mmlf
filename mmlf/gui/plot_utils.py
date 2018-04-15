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

""" Some helper function for viewers and graphics logging."""

import numpy

from mmlf.framework.state import State

def generate2dStateSlice(varyDimensions, stateSpace, defaultDimValues,
                         gridNodesPerDim, varyValueRanges=None):
    """ Generate a set of states that form a 2d slice through state space.
    
    The set of states consists of gridNodesPerDim**2 states. Each of them has 
    the value given by defaultDimValues except for the values in the two 
    dimensions passed by varyDimensions. In these two dimensions, the value
    is determined based on a 2d grid that fills [0,1]x[0,1].
    """
    assert len(varyDimensions) == 2, \
        " We need two varyDimensions to create a 2d state space slice."    
    
    assert (sorted(stateSpace.keys()) == sorted(set(defaultDimValues.keys() + varyDimensions))), \
        "Cannot create a 2d state space slice since value definition is not "\
        "consistent with state space definition."
    
    if varyValueRanges == None: # Set default value ranges
        varyValueRanges = [stateSpace[varyDimensions[0]].getValueRanges()[0],
                           stateSpace[varyDimensions[1]].getValueRanges()[0]]
    
    # Sort state dimensions according to dimension name
    dimensions = [stateSpace[dimName] for dimName in sorted(stateSpace.keys())]
    defaultValue = [defaultDimValues[dimName] for dimName in sorted(defaultDimValues.keys())]
    
    # The indices of the dimensions which vary
    varyDimensionIndex1 = sorted(stateSpace.keys()).index(varyDimensions[0])
    varyDimensionIndex2 = sorted(stateSpace.keys()).index(varyDimensions[1])
    
    # Create the 2d slice
    slice2d = {}
    for i, value1 in enumerate(numpy.linspace(varyValueRanges[0][0],
                                              varyValueRanges[0][1],
                                              gridNodesPerDim)):
        for j, value2 in enumerate(numpy.linspace(varyValueRanges[1][0],
                                                  varyValueRanges[1][1],
                                                  gridNodesPerDim)):
            # instantiate default value
            defaultValue[varyDimensionIndex1] = value1
            defaultValue[varyDimensionIndex2] = value2
            # Create state object
            slice2d[(i,j)] = State(defaultValue, dimensions)
            
    return slice2d

def generate2dPlotArray(function, stateSlice, continuousFunction, shape):
    """ Generate a 2d array of function values based on a state slice.
    
    For each state of the stateSlice, the corresponding value of function is 
    determined (function being a function from state space to real numbers). 
    These values are stored than in a 2d array that can be used e.g. for plotting.
    If continuousFunction is True, the function can take on continuous values,
    otherwise discrete ones.
    """
    # The plot raster
    values = numpy.ma.array(numpy.zeros(shape), mask = numpy.zeros(shape))
    
    colorValue = 0
    colorMapping = {}
    for (i,j), state in stateSlice.iteritems():
        # Evaluate function for this state
        functionValue = function(state)
    
        # Deal with situations where the function is only defined over 
        # part of the state space
        if functionValue == None or functionValue in [numpy.nan, numpy.inf, -numpy.inf]:
            values.mask[i,j] = True
            continue
        if not continuousFunction:
            # Determine color value for function value
            if not functionValue in colorMapping:
                # Choose value for function value that occurs for the 
                # first time
                colorMapping[functionValue] = colorValue
                colorValue += 1
            values[i,j] = colorMapping[functionValue]
        else:
            if hasattr(functionValue, '__iter__'):
                values[i,j] = functionValue[0] # TODO: Currently only one dimension
            else:
                values[i,j] = functionValue
            
    return values, colorMapping
            