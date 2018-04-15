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
""" A black-box optimization algorithm based on evolution strategies. """

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"

import random
from numpy import zeros, diag, all, array, inf, sqrt
from numpy.linalg import norm
from numpy.random import multivariate_normal

from mmlf.resources.optimization.optimizer import Optimizer

class ESOptimizer(Optimizer):
    """ A black-box optimization algorithm based on evolution strategies

    This black-box optimization algorithm is based on a simple evolution
    strategies (ES). The mu+lambda ES maintains a set of mu potential 
    parents and generates for each generation a set of lambda children. These 
    children are obtained by mutating randomly sampled parents. This mutation
    is accomplished by adding normally distributed 0-mean random values to the
    individuals.
    
    After the children have been evaluated, parents and children are merged in a  
    set and the mu fittest individuals in this set form the new parent set.
    The mutation standard deviation  is adjusted based on Rechenbergs 1/5 rule.
    
    Mandatory parameters:
     * sampleInstance: A method which returns a valid parameter vector. This 
                       method can always return the same instance. However,
                       it is more common to provide a method that returns
                       stochastically sampled different individuals. The 
                       returned individuals might for instance be used to
                       generate an initial population. 
    
    **CONFIG DICT**
        :sigma: : The initial standard deviation of the the mutation operator. sigma can be either a single value (meaning that each dimension of the parameter vector will be equally strong mutated) or a vector with the same shape as the parameter vector. 
        :populationSize: : The number of parents that survive at the end of a generation (i.e. mu)
        :evalsPerIndividual: : The number of fitness values that the optimizer expects for each individual
        :numChildren: : The number of children evaluated within a generation (i.e. lambda).
    """
    DEFAULT_CONFIG_DICT = {'sigma':  1.0,
                           'populationSize' : 5,
                           'evalsPerIndividual': 1,
                           'numChildren': 10}
    
    def __init__(self, sampleInstance, sigma=1.0, populationSize=5, 
                 evalsPerIndividual=1, numChildren=10, **kwargs):
        super(ESOptimizer, self).__init__(populationSize, evalsPerIndividual)
        
        self.sigma = sigma
        self.sampleInstance = sampleInstance
        self.numChildren = numChildren
        
        ## NOTE: We have to overwrite this variable set by superclass since
        #        the first generation is larger than populationSize
        # The current fitness estimates of the population
        self.fitnesses = [0.0 for i in range(populationSize + numChildren)]
        # Counts for how many times the individuals have been evaluated
        self.evalsCounter = [0 for i in range(populationSize + numChildren)]
        
        # Start with a randomly sampled set of children
        self.children = [list(self.sampleInstance()) 
                            for i in range(populationSize + numChildren)]
        
        # A mapping from fitness to individual with the shape
        # [(fitness_1, individual_1), ..., (fitness_n, individual_n)]
        self.fitnessOfParents = []
        
        # Replace sigma by a vector [sigma, ..., sigma] if necessary
        if isinstance(self.sigma, float):
            self.sigma = array([self.sigma for i in range(len(self.children[0]))])
        self.initialSigma = array(self.sigma) #Copy for reinitializing
        
        # Remember the best individual found so far    
        self.bestIndividual = None
        self.maxFitness = -inf

        
    def getActiveIndividual(self):
        """ Returns the parameter vector of the currently active individual"""
        return self.children[self.activeIndividualIndex]

    def getAllIndividuals(self):
        """ Returns all individuals of the current population """
        return self.children  
    
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
        
        if all(array(self.evalsCounter) > self.evalsPerIndividual):
            self._createNextGeneration(self.children, self.fitnesses)
        elif self.evalsCounter[self.activeIndividualIndex] == self.evalsPerIndividual:       
            # Activate the next individual
            self.activeIndividualIndex += 1
            if self.activeIndividualIndex >= len(self.children):
                self._createNextGeneration(self.children, self.fitnesses)

        return False

    def getBestIndividual(self):
        """ Returns the parameter vector of the best individual found so far"""
        if self.bestIndividual is not None:
            return self.bestIndividual
        else:
            # We have not yet finished a generation
            maxFitness = -inf
            bestIndividual = None
            for individual, invidualFitness in zip(self.children, self.fitnesses):
                if invidualFitness > maxFitness:
                    bestIndividual = individual
                    maxFitness = invidualFitness
            return bestIndividual

    def getLastGenerationsBestIndividual(self):
        """ Returns the best individual of the last generation.
        
        Returns the best individual of the last generation that has been fully 
        evaluated. This method might be favorable in non-stationary environments
        that change over time.
        """
        if self.fitnessOfParents == []:
            # We are within the first generation, so the fittest individual is
            # not yet a parent
            return sorted(zip(self.fitnesses, self.children))[-1][1]
        else:
            return sorted(self.fitnessOfParents)[-1][1]
    
    def reinitialize(self):
        """ """
        # Restore initial sigma value
        self.sigma = array(self.initialSigma)
        
        # We seed the new evolution with the results of the last
        initialWeights = self.getLastGenerationsBestIndividual()
        # Rescale weights
        initialWeights = initialWeights / norm(initialWeights) * self.sigma * 5 * sqrt(len(initialWeights))
                
        # Start with a randomly sampled set of children
        self.children = [self._mutate(initialWeights)
                            for i in range(self.populationSize + self.numChildren)]
        
        # The current fitness estimates of the population
        self.fitnesses = [0.0 for i in range(self.populationSize + self.numChildren)]
        # Counts for how many times the individuals have been evaluated
        self.evalsCounter = [0 for i in range(self.populationSize + self.numChildren)]
        
        # A mapping from fitness to individual with the shape
        # [(fitness_1, individual_1), ..., (fitness_n, individual_n)]
        self.fitnessOfParents = []
        
        # Remember the best individual found so far    
        self.bestIndividual = None
        self.maxFitness = -inf
        
    
    def _createNextGeneration(self, individuals, fitness):
        # Check if one of the tested individuals achieved a higher fitness than 
        # all other individuals before       
        for individual, invidualFitness in zip(individuals, fitness):
            if invidualFitness > self.maxFitness:
                self.bestIndividual = individual
                self.maxFitness = invidualFitness
                
        # Put children and parents into one set        
        self.fitnessOfParents.extend(zip(fitness,
                                         individuals))
        
        # Select self.populationSize fittest individuals
        self.fitnessOfParents = sorted(self.fitnessOfParents)[-self.populationSize:]
        parents = map(lambda x: x[1], self.fitnessOfParents)
        
        # Determine how many children from the last generation survived
        survivors = 0
        for child in individuals:
            if child in parents:
                survivors += 1
        survivorRate = float(survivors)/len(individuals)
        
        # According to Rechenberg's 1/5 rule, sigma is increased when 
        # survivorRate > 0.2 and decreased when survivorRate < 0.2
        if survivorRate > 0.2:                  
            self.sigma *= 1.5
        elif survivorRate < 0.2:                  
            self.sigma *= 0.666
        
        # Create children
        self.children = [self._mutate(random.choice(parents)) 
                                for i in range(self.numChildren)]
        
        # Reset some attributes
        self.activeIndividualIndex = 0
        self.evalsCounter = [0 for i in range(self.numChildren)]
        self.fitnesses = [0.0 for i in range(self.numChildren)]
    
    def _mutate(self, individual):    
        mutation = multivariate_normal(zeros(len(individual)), diag(self.sigma))
        return list(array(individual) + mutation)
  