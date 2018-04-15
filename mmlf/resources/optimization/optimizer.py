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
""" Interface for MMLF black box optimization algorithms

MMLF optimizers must implement the following methods:
 * getActiveIndividual
 * getAllIndividuals
 * tellFitness
 * tellAllFitness
 * getBestIndividuals
"""

__author__ = "Jan Hendrik Metzen"
__copyright__ = "Copyright 2011, University Bremen, AG Robotics"
__credits__ = ['Mark Edgington']
__license__ = "GPLv3"
__version__ = "1.0"
__maintainer__ = "Jan Hendrik Metzen"
__email__ = "jhm@informatik.uni-bremen.de"


class Optimizer(object):
    """ Interface for MMLF black box optimization algorithms

    MMLF optimizers must implement the following methods:
     * getActiveIndividual():  Returns the parameter vector of the currently
                               active individual
     * getAllIndividuals(): Returns all individuals of the current population
     * tellFitness(fitnessSample): Gives one fitness sample for the currently 
                                   active individual
     * tellAllFitness(fitness, individuals): Gives fitness samples for all given
                                            individuals
     * getBestIndividual(): Returns the parameter vector of the best individual 
                            found so far
    """
    
    # A dictionary that will contain a mapping from optimizer name
    # to the respective optimizer class
    OPTIMIZER_DICT= None
    
    def __init__(self, populationSize, evalsPerIndividual):
        self.populationSize = populationSize
        self.evalsPerIndividual = evalsPerIndividual
        # The current fitness estimates of the population
        self.fitnesses = [0.0 for i in range(populationSize)]
        # Counts for how many times the individuals have been evaluated
        self.evalsCounter = [0 for i in range(populationSize)]
        # The index of the currently active individual
        self.activeIndividualIndex = 0
            
    def getActiveIndividual(self):
        """ Returns the parameter vector of the currently active individual"""
        raise NotImplementedError("Optimizer is only an interface")
  
    def getAllIndividuals(self):
        """ Returns all individuals of the current population """
        raise NotImplementedError("Optimizer is only an interface")
    
    def tellFitness(self, fitnessSample, individual=None):
        """ Gives one fitness sample for the individual
        
        Provides the fitness *fitnessSample* obtained in one evaluation of
        the given individual. In individual==None, then the fitnessSample is 
        attributed to the currently active individual.
        """
        raise NotImplementedError("Optimizer is only an interface")

    def tellAllFitness(self, individuals, fitness):
        """ Gives fitness samples for all given individuals
        
        Gives fitness samples *fitness* for all given *individuals*. It is
        assumed that individuals is actually the whole population. A call of 
        this method may trigger the creation of the next generation.        
        """
        self._createNextGeneration(individuals, fitness)

    def getBestIndividual(self):
        """ Returns the parameter vector of the best individual found so far"""
        raise NotImplementedError("Optimizer is only an interface")

    @staticmethod
    def getOptimizerDict():
        """ Returns dict that contains a mapping from optimizer name to optimizer class. """
        # Lazy initialization  
        if Optimizer.OPTIMIZER_DICT == None:
            from mmlf.resources.optimization.cmaes_optimizer import CMAESOptimizer
            from mmlf.resources.optimization.evolution_strategy import ESOptimizer
            from mmlf.resources.optimization.random_search import RandomSearchOptimizer
             
            # A static dictionary containing a mapping from name to class 
            Optimizer.OPTIMIZER_DICT = {'cmaes': CMAESOptimizer,
                                        'evolution_strategy': ESOptimizer,
                                        'random_search': RandomSearchOptimizer}
            
        return Optimizer.OPTIMIZER_DICT

    @staticmethod
    def create(optimizerSpec, sampleInstance):
        """ Factory method that creates optimizer based on spec-dictionary. """
        # Determine the model class
        if optimizerSpec['name'] in Optimizer.getOptimizerDict():
            OptimizerClass = Optimizer.getOptimizerDict()[optimizerSpec['name']]
        else:
            raise NotImplementedError("Optimization type %s unknown" % optimizerSpec["name"])
        
        return OptimizerClass(sampleInstance=sampleInstance,
                              **optimizerSpec)
