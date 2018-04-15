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
# Created: 2009/11/23
""" The CMA-ES black-box optimization algorithm

Module that contains the covariance matrix adaptation - evolution strategy 
(CMA-ES) black-box optimization algorithm.

See also:
Nikolaus Hansen and Andreas Ostermeier, 
"Completely derandomized self-adaptation in evolution strategies",
Evolutionary Computation 9 (2001): 159--195.
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
import numpy

from mmlf.resources.optimization.optimizer import Optimizer
from mmlf.resources.evolutionary_algorithms.cmaes import CMAEvolutionStrategy

class CMAESOptimizer(Optimizer):
    """ The CMA-ES black-box optimization algorithm

    Optimizer that uses CMA-ES for black-box optimization
    
    Optional parameters:
      * sampleInstance: A method which returns a valid parameter vector. This 
                       method can always return the same instance. However,
                       it is more common to provide a method that returns
                       stochastically sampled different individuals. The 
                       returned individuals might for instance be used to
                       generate an initial population. 
                       
    **CONFIG DICT**
        :evalsPerIndividual: : The number of fitness values that the optimizer expects for each individual.
        :numParameters: : The number of parameters (dimensionality of parameter vector). If None, an initial parameter vector must be specified. Otherwise the initial parameters are drawn randomly. 
        :initialMean: : The initial parameter vector.  
        :sigma0: : The initial standard deviation of the search distribution.
    """
    
    DEFAULT_CONFIG_DICT = {'evalsPerIndividual':  1,
                           'initialMean' : None,
                           'sigma': 1.0}
    
    def __init__(self, sampleInstance=None, evalsPerIndividual=1,
                  initialMean=None, sigma=1.0, **kwargs):        
        if initialMean == None:
            assert (sampleInstance != None), ("If no initial mean is provided, "\
                                              "a method for sampling instances "\
                                              "must be provided")
            initialMean = sampleInstance()
            
        self.initialMean = initialMean
        self.sigma = sigma
        
        # The metaheuristic CMA-ES which is used to optimize
        # the parameters        
        self.cmaes = CMAEvolutionStrategy(x0=initialMean, sigma0=sigma)
        # Holds the current population of parameters
        # which are optimized by CMA-ES
        self.population = map(list, self.cmaes.Ask())
        self.lastPopulation =  None
        
        super(CMAESOptimizer, self).__init__(len(self.population),
                                             evalsPerIndividual)
        
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
             
        # Since CMA-ES minimizes, we accumulate the ->negative<- of the fitnessSample  
        self.fitnesses[individualIndex] -= fitnessSample
        self.evalsCounter[individualIndex] += 1
        
        if numpy.all(numpy.array(self.evalsCounter) > self.evalsPerIndividual):
            self._createNextGeneration()
        elif self.evalsCounter[self.activeIndividualIndex] == self.evalsPerIndividual:       
            # Activate the next individual
            self.activeIndividualIndex += 1
            if self.activeIndividualIndex >= len(self.population):
                self._createNextGeneration()

        return self.cmaes.stop

    def getBestIndividual(self):
        """ Returns the parameter vector of the best individual found so far"""
        best = self.cmaes.Best()[0] # Only return the parameters not the fitness!
        # If we have not yet finished the first generation
        if best == []:
            best = random.choice(self.population)
        return best
    
    def getLastGenerationsBestIndividual(self):
        """ Returns the best individual of the last generation.
        
        Returns the best individual of the last generation that has been fully 
        evaluated. This method might be favorable in non-stationary environments
        that change over time.
        """
        if self.lastPopulation is not None:
            return min(self.lastPopulation)[1]
        else:
            return random.choice(self.population)

    def reinitialize(self):
        self.cmaes = CMAEvolutionStrategy(x0=self.getLastGenerationsBestIndividual(),
                                          sigma0=self.sigma)

    def _createNextGeneration(self):
        # Inform CMA-ES about the fitness
        self.cmaes.Tell(self.population, self.fitnesses)
        # Remember last generation and its fitness
        self.lastPopulation = zip(self.fitnesses, self.population)
         
        # We request the next weight-population from CMA-ES
        self.population = map(list, self.cmaes.Ask())
        self.activeIndividualIndex = 0
        self.evalsCounter = [0 for ind in self.population]
        self.fitnesses = [0.0 for ind in self.population]
    