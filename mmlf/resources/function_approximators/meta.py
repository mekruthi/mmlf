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
""" Several meta function approximators based on more simple ones like CMAC. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"
    
from mmlf.resources.function_approximators.function_approximator import FunctionApproximator

class MultiFunctions(FunctionApproximator):
    """ Meta function approximator based on the multifunction principle. 
    
	The Multifunction is a meta function approximator that uses a set of 
	simple function approximators where each of them is responsible for a 
	subproblem (e.g. a subarea or a region)
	It's current implementation is specific for BRIO:  
    The BRIO board is subdivided following a given scheme, and to every area 
    a function approximator is assigned
    
    Scheme 1: manual subdivision
		 
    How can I use this Multifunction class?
       first create your instance 
       1) MF= MultiFunctions(myFunctionAproximator, configDict)
          **It is expected that your (my)FunctionAproximator has a constructor with
          configDict as argument**
       2) MF.createMultiFunctions(mySubdivisionShema)
          ** the default scheme is a manual subdivision of the BRIO board**
    """
    
    DEFAULT_CONFIG_DICT = {}
    
    SUPPORTED_STATE_SPACES = ["DISCRETE", "CONTINUOUS"]
    
    def __init__(self, stateSpace, actions, subDivisionScheme,
                 subFunctionApproximator, **kwargs):
        super(MultiFunctions, self).__init__(stateSpace)
        
        self.subFunctionApproximator = subFunctionApproximator 
        
        self.stateSpace = stateSpace
        self.actions = actions       
        
        self.numberOfInputs = len(stateSpace.keys())
        self.numberOfOutputs = len(actions)
        
        # Determine the sub function approximator class
        self.approximatorClass = None
        if subFunctionApproximator["name"] == 'QCON':
            from mmlf.resources.function_approximators.qcon import QCON
            self.approximatorClass = QCON
        elif subFunctionApproximator["name"] == 'CMAC':
            from mmlf.resources.function_approximators.cmac import CMAC
            self.approximatorClass = CMAC
            for dimName, dimDescr in stateSpace.iteritems():
                if dimName in ["ballPositionX", "ballPositionY"]:
                    dimDescr["resolution"] /= 3

        # Create the subdivision of the game
        if subDivisionScheme == 'manual':
            self.areas = MultiFunctions._manualSubdivide()
                
        # A mapping form area to function approximator
        self.areaApproximator = self._assignApproximatorToArea()
        
        self.boardWidth = 0.276
        self.boardHeight = 0.2305
    
    @staticmethod
    def _manualSubdivide():
        """
        First we implement a manual subdivision of the BRIO board  into 
        areas
        Every area is presented through a set of regions 
        Every region corresponds to a rectangle and 
        can be specified  by  the coordinates of the lower left (xi1,yi1)
        and upper right point (xi2,yi2) of the rectangle. A region is 
        coded as ((xi1,xi2), (yi1,yi2))
        
        area: (( (x11,x12), (y11,y12)), ..., ( (xi1,xi2), (yi1,yi2)))
        region : ((xi1, xi2), (yi1,yi2))
        """
        # Define all areas (consisting potentially of several subregions)
        # Unit: pixels (?)
        # Origin: left lower corner of the image
        pixel_areas = [
            (((62,108),(257,275)),((30,62),(243,275))),#1
            (((0,30),(228,275)),),#2
            (((0,30),(175,228)),),#3
            (((0,30),(125,175)),),#4
            (((30,62),(125,175)),),#5
            (((30,75),(175,197)), ((62,75),(165,175))),#6
            (((30,75),(197,225)),),#7
            (((30,75),(225,243)), ((62,75),(243,257))),#8
            (((75,108),(210,257)),),#9
            (((75,108),(165,210)),),#10
            (((62,108),(125,165)),),#11
            (((108,139),(125,172)),),#12
            (((139,184),(125,160)),),#13
            (((154,184),(160,185)), ((154,169),(185,219))),#14
            (((139,154),(160,207)), ((108,139),(172,195))),#15
            (((108,139),(195,225)),),#16
            (((139,154),(207,243)), ((108,139),(225,243))),#17
            (((108,154),(243,275)),),#18
            (((154,184),(219,275)), ((169,184),(185,219)))#19
            ]
            
        #scale the areas:
        boardWidth = 0.276
        boardHeight = 0.2305
        minX = 0  #image x0
        maxX = 184 
        minY = 125
        maxY = 275
        '''
        maxY
         ^
         |
         |
        minY
        minX----->maxX
        '''
        mappingX = lambda x: (float(x) - minX) / (maxX - minX) * boardWidth - boardWidth/2
        mappingY = lambda y: (float(y) - minY) / (maxY - minY) * boardHeight - boardHeight/2
        areas = []  
        # For all areas
        for pixel_area in pixel_areas:
            area = []
            # For all subregions of this area 
            for pixel_region in pixel_area:
                # Scale the region
                region = ((mappingX(pixel_region[0][0]),
                           mappingX(pixel_region[0][1])),
                          (mappingY(pixel_region[1][0]),
                           mappingY(pixel_region[1][1])))
                # Append to the area
                area.append(region)
            # Append the constructed region to the list of regions
            areas.append(tuple(area))
            
        return areas
                
    def _assignApproximatorToArea(self):
        """
        Assign to each area a function approximator  with the given config
        """
        return dict((area, self.approximatorClass(self.stateSpace,
                                                  self.actions,
                                                  **self.subFunctionApproximator))
                                     for area in self.areas)
                            
    
    def getArea(self, state):
        """
        Get the area in which the given position state falls 
        """        
        # Compute the x and y position in board coordinates
        posX = state[self.stateSpace.keys().index("ballPositionX")]
        minX, maxX = self.stateSpace["ballPositionX"]['dimensionValues'][0]
        posX =  posX * (maxX - minX) + minX
        
        posY = state[self.stateSpace.keys().index("ballPositionY")]
        minY, maxY = self.stateSpace["ballPositionY"]['dimensionValues'][0]
        posY =  posY * (maxY - minY) + minY
        
        # Clip to board region
        posX = min(max(posX, -self.boardWidth/2), self.boardWidth/2)
        posY = min(max(posY, -self.boardHeight/2), self.boardHeight/2)
        
        for aIndex, area in enumerate(self.areas):
            for rIndex, region in enumerate(area):
                if region[0][0] <= posX <= region[0][1] \
                      and region[1][0] <= posY <= region[1][1]:
#                    print "Area %s/Region %s responsible for point (%s,%s)." \
#                                % (aIndex, rIndex, posX, posY)
                    return area
                
        raise Exception("The given position (%s,%s) does not belong to any area!" % (posX, posY))              
    
    def computeQ(self,state, action):
        """
        Computes the Q-value of the given state, action pair
        
        Compute the Q-value by calling the computeQ routine
        of the function approximator responsible for the area in that
        the given state falls
        """
        # Determine the responsible approximator
        respApproximator = self.areaApproximator[self.getArea(state)]
        # Let the responsible approximator compute the Q-function                      
        return respApproximator.computeQ(state, action)
    
    def train(self, trainingSet):
        """
        Train all function approximator on samples from the training
        set that are of relevance for them
        """
        #print "The training set %s" % trainingSet
        
        trainingSetPartition = dict()
        
        # Split the training set according to areas
        for (state, action), target in trainingSet.iteritems():
            # Get the information about the area 
            # from the state action descriptor
            area = self.getArea(state)
            
            trainingSetPartition[area] = trainingSetPartition.setdefault(area, {})
            trainingSetPartition[area][(state, action)] = target
        
        # Train each function approximator on its respective training set
        for area, partialTrainingSet in trainingSetPartition.iteritems():
            self.areaApproximator[area].train(partialTrainingSet)
    
    @staticmethod
    def drawAreas(areas):
        import numpy as np
        import matplotlib.pyplot as plt
        
        # num_a=np.zeros([len(self.areas),4,2,2])
        color='r'
        for area in areas: # a for area
            for region in area: # r for region
                x = np.array([region[0][0],region[0][1],region[0][1],region[0][0]])
                y = np.array([region[1][0],region[1][0],\
                                  region[1][1],region[1][1]])
                plt.fill(x,y,color,alpha=0.5)
            if color =='r':
                color='b'
            elif color=='b':
                color='r' 
            plt.draw()
#            import time
#            time.sleep(1)
#            #raw_input()
                            
        plt.grid(True)
        plt.show()

        
if __name__ == "__main__":
    import pylab
    from mmlf.worlds.brio.environments.LabyrinthSpecParser import LabyrinthSpecParser
    
#    pylab.ion()
    
    labyrinthSpecParser = LabyrinthSpecParser("/home/jmetzen/Repositories/mmlf_development/mmlf/worlds/brio/environments/line_points.txt")
    subgoals = labyrinthSpecParser.parseSubgoals()
    holes = labyrinthSpecParser.parseHoles()
    
    subgoalsX = []
    subgoalsY = []
    for subgoal in subgoals:
        subgoalsX.append(subgoal[0])
        subgoalsY.append(subgoal[1])
    
    pylab.plot(subgoalsX, subgoalsY)
    
    for hole in holes:
        print hole
        pylab.plot([hole[0]], [hole[1]], "yo")
    
    areas = MultiFunctions._manualSubdivide()
    MultiFunctions.drawAreas(areas)
    pylab.show()
    
    
   
