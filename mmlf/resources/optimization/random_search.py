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
# Created: 2010/01/11
""" A black-box optimization algorithm based on random search. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import numpy

from mmlf.resources.optimization.optimizer import Optimizer

class RandomSearchOptimizer(Optimizer):
    """ The random search optimization algorithm

    Optimizer that uses a random search heuristic for black-box optimization.
    This optimization algorithm does not search the search space "intelligent" 
    but just randomly samples candidate solutions uniformly from the search  
    space and test these. It is mainly useful as a benchmark for more  
    sophisticated search strategies.

    Mandatory parameters:
     * sampleInstance: A method which returns a valid parameter vector. This 
                       method can always return the same instance. However,
                       it is more common to provide a method that returns
                       stochastically sampled different individuals. The 
                       returned individuals might for instance be used to
                       generate an initial population. 

    **CONFIG DICT**
        :populationSize: : The number of individuals evaluated in parallel. Since the search distribution does not change over time, this can be set to any arbitrary value between 1 and maxEvals for this optimizer.
        :evalsPerIndividual: : The number of fitness values that the optimizer expects for each individual.
    """
    
    DEFAULT_CONFIG_DICT = {'populationSize' : 10,
                           'evalsPerIndividual': 1}
    
    def __init__(self, sampleInstance, populationSize=10, evalsPerIndividual=1,
                 **kwargs):
        super(RandomSearchOptimizer, self).__init__(populationSize,
                                                    evalsPerIndividual)
        
        self.sampleInstance = sampleInstance
        
        self.population = [self.sampleInstance()
                             for i in range(self.populationSize)]
        
        self.bestIndividual = None
        self.maxFitness = -numpy.inf
        
    def getActiveIndividual(self):
        """ Returns the parameter vector of the currently active individual"""
        return self.population[self.activeIndividualIndex]

    def getAllIndividuals(self):
        """ Returns all individuals of the current population """
        return self.population  
    
    def tellFitness(self, fitnessSample, individual=None):
        """ Gives one fitness sample for the individual
        
        Provides the fitness *fitnessSample* obtained in one evaluation of
        the given individual. In individual==None, then the fitnessSample is 
        attributed to the currently active individual.
        """
        # Determine index of individual to which the fitnessSample will be
        # attributed
        if individual == None:
            individualIndex = self.activeIndividualIndex
        else:
            raise Exception("This case is not yet implemented") #TODO
             
        # Accumulate fitness samples  
        self.fitnesses[individualIndex] += fitnessSample
        self.evalsCounter[individualIndex] += 1
        
        if numpy.all(numpy.array(self.evalsCounter) > self.evalsPerIndividual):
            self._createNextGeneration(self.population, self.fitnesses)
        elif self.evalsCounter[self.activeIndividualIndex] == self.evalsPerIndividual:       
            # Activate the next individual
            self.activeIndividualIndex += 1
            if self.activeIndividualIndex >= len(self.population):
                self._createNextGeneration(self.population, self.fitnesses)

        return False

    def getBestIndividual(self):
        """ Returns the parameter vector of the best individual found so far"""
        return self.bestIndividual
    
    def getLastGenerationsBestIndividual(self):
        """Returns the best individual of the last generation."""
        # Random search has no generations, we return the overall best 
        # individual
        return self.getBestIndividual()
    
    def _createNextGeneration(self, individuals, fitness):
        # Check if one of the tested individuals achieved a higher fitness than 
        # all other individuals before       
        for individual, invidualFitness in zip(individuals, fitness):
            if invidualFitness > self.maxFitness:
                self.bestIndividual = individual
                self.maxFitness = invidualFitness
        
        # Create a new population sampled randomly. 
        self.population = [self.sampleInstance()
                             for i in range(self.populationSize)]
        
        # Reset some attributes
        self.activeIndividualIndex = 0
        self.evalsCounter = [0 for i in range(self.populationSize)]
        self.fitnesses = [0.0 for i in range(self.populationSize)]
    