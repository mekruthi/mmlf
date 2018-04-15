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

""" cma.py

 A stochastic optimizer Covariance Matrix Adaptation Evolution Strategy,
    CMA-ES, for robust non-linear non-convex function minimization

 CMA-ES searches for a minimizer (a solution x in R**n) of an
 objective function f (cost function), such that f(x) is
 minimal. Regarding f, only function values for candidate solutions
 need to be available, gradients are not necessary. Even less
 restrictive, only a passably reliably ranking of the candidate
 solutions in each iteration is necessary. 

 Two interfaces are provided: 
 
 function fminCMA(func,x0,sigma0,...) -- runs a complete
    minimization of the objective function func. 

 class CMAEvolutionStrategy -- allows for minimization such that
     the control of the iteration loop remains with the user. 

 Used packages: numpy, time, sys, string, pprint, pygsl.eigen, pylab (for PlotCMAdata())
 
"""

#
# Author: Nikolaus Hansen, copyright 2008.
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License, version 2,
#    as published by the Free Software Foundation.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 

# TODO re-consider interface:
# interface for PlotCMA : interactive should be parameter, such that the plot can be seen
# split into "parameters" and "options", where the latter can be changed any-time
#    by reading them from a (properties) file, see myproperties.py
# 
# typical parameters in scipy.optimize: disp, xtol, ftol, maxiter, maxfun, callback=None
#         maxfev, diag (A sequency of N positive entries that serve as 
#                 scale factors for the variables.)
#           full_output -- non-zero to return all optional outputs.
#   If xtol < 0.0, xtol is set to sqrt(machine_precision)
#    'infot -- a dictionary of optional outputs with the keys:
#                      'nfev' : the number of function calls...
# 
# consider resume functionality 
#                 
#    see eg fmin_powell 
# typical returns
#        x, f, dictionary d 
#        (xopt, {fopt, gopt, Hopt, func_calls, grad_calls, warnflag}, <allvecs>)
# 
# TODO eig(): revise check for equal eigenvectors, quotient between adjacent eigenvalues.
#    idx = diagD[1:]/diagD[0:N] < 1.05
#
# TODO solve problem with show(): use testcma.py: graphic does not show up, but show()
#   is needed. There should be another function which does not enter interactive mode. 
#   
# TODO: import should rather be local in general!?
# TODO (later): implement readSignals from a file like properties file (to be called after tell())

import time      # from time import asctime, time 
import sys       # from sys import stdout
import string    # from string import join, split
import numpy     # from numpy import eye, zeros, linalg.eig, sort, argsort, random, ones,... 
from numpy import inf, array, log, sqrt, arange, dot, sum, outer, cos, pi, floor, Inf
import random

# sys.py3kwarning = True  # TODO: out-comment from version 2.6

# from math import log, sqrt

# changes:
# 08/10/01: option evalparallel introduced,
#           bug-fix for scaling being a vector 
# 08/09/26: option CMAseparable becomes CMAdiagonal
__version__="0.9.00" 

class _Struct(object):
    pass

#____________________________________________________________
#____________________________________________________________
#   
class _GenoPheno(object):
    """Genotype-phenotype transformation for convenient scaling.
    """
    def pheno(self, x):
        y = array(x)
        if self.scales is not 1:  # just for efficiency
            y = self.scales * y
            # print 'transformed to phenotyp'
        if self.typical_x is not 0:
            y = y + self.typical_x
        return y
    def geno(self, y):
        x = array(y)
        if self.typical_x is not 0:
            x = x - self.typical_x
        if self.scales is not 1:  # just for efficiency
            x = x / self.scales  # for each element in y
        return x

    def __init__(self, scaling=None, typical_x=None):
        if numpy.size(scaling) > 1 or scaling:  # 
            self.scales = scaling  # CAVE: is not a copy
        else:
            self.scales = 1

        if typical_x:
            self.typical_x = typical_x
        else:
            self.typical_x = 0
        # print self.typical_x # TODO: needs to be tested !? 
        # print self.scales

#____________________________________________________________
#____________________________________________________________
# 
class CRotation(object):
    """Rotation class that implements an orthogonal linear transformation, one for each
    dimension. Used to implement non-separable test functions. 

    :Example:
       import numpy, cma
       R = cma.CRotation()
       R2 = cma.CRotation() # another rotation
       x = numpy.array((1,2,3))
       print R(x)
       print R2(x)
       print R(x)
       print R(R(x), inverse=1)

       [ 0.02609877  3.72521124  0.34945679]
       [ 0.38345182  3.70862509  0.3147454 ]
       [ 0.02609877  3.72521124  0.34945679]
       [ 1.  2.  3.]
    """
    dicMatrices = {} 
    def __init__(self):
        self.dicMatrices = {} # otherwise there might be shared bases which is
                              # probably not what we want 
    def __call__(self, x, inverse=False): # function when calling an object
        """Rotates the input array x with a fixed rotation matrix (dicMatrices['str(len(x))'])
        """
        N = x.shape[0]  # can be an array or matrix, TODO: accept also a list of arrays?
        if not self.dicMatrices.has_key(str(N)): # create new N-basis for once and all
            B = numpy.random.randn(N, N)
            for i in xrange(N):
                for j in xrange(0, i):
                    B[i] -= dot(B[i], B[j]) * B[j]
                B[i] /= sqrt(sum(B[i]**2))
            self.dicMatrices[str(N)] = B
        if inverse: 
            return dot(self.dicMatrices[str(N)].T, x)  # compute rotation
        else: 
            return dot(self.dicMatrices[str(N)], x)  # compute rotation

# Use Rotate(x) to rotate x 
Rotate = CRotation()

def frot(x, fun, rot=1):
    if len(shape(array(x))) > 1:  # parallelized
        res = []
        for x in x:
            res.append(frot(x, fun, rot))
        return res

    if rot:
        return fun(Rotate(x))
    else:
        return fun(x)

def frand(x): 
    """Random test objective function"""
    return numpy.random.random(1)[0]

def flinear(x):
    return -x[0]

def fsphere(x):
    """Sphere (squared norm) test objective function"""
    return sum(x**2)

def normalSkew(f):
    N = numpy.random.randn(1)[0]**2
    if N < 1:
        N = f * N  # diminish blow up lower part
    return N

def fnoiseC(x, func=fsphere, fac=10, expon=0.8):
    f = func(x)
    N = numpy.random.randn(1)[0]/numpy.random.randn(1)[0]
    return max(1e-19, f + (float(fac)/len(x)) * f**expon * N)

def fnoise(x, func=fsphere, fac=10, expon=1):
    from numpy import exp, log, random, log10
    f = func(x)
    sig = float(fac)/float(len(x))
    R = random.randn(1)[0]
    R = log10(f) + expon * abs(10-log10(f)) * random.rand(1)[0]
    # R = log(f) + 0.5*log(f) * random.randn(1)[0] 
    # return max(1e-19, f + sig * (f**log10(f)) * exp(R)) 
    # return max(1e-19, f * numpy.exp(sig * N / f**expon)) 
    # return max(1e-19, f * normalSkew(f**expon)**sig)
    return f + 10**R  # == f + f**(1+0.5*RN)

def fcigar(x, rot=0):
    """Cigar test objective function"""
    if rot:
        x = Rotate(x)  
    return x[0]**2 + 1e6 * sum(x[1:]**2)
def felli(x, rot=0, par=False):
    """Ellipsoid test objective function"""
    if par:  
        f = []
        for xi in x:
            f.append(felli(xi, rot))
        return f
    if rot:
        x = Rotate(x)  
    N = len(x)
    return sum((10**(3.*arange(N)/(N-1.))*x)**2)
def frosen(x):  
    """Rosenbrock test objective function"""
    return sum(100.*(x[:-1]**2-x[1:])**2 + (1.-x[:-1])**2)
def frastrigin(x):
    """Rastrigin test objective function"""
    N = len(x)
    return 10*N + sum(x**2 - 10*cos(2*pi*x))

#____________________________________________________________
#____________________________________________________________
# 
def PrintOptions(): 
    """ pretty-prints out options calling GetOptions()
    """
    for i in sorted(GetOptions().items()): print i[0].rjust(45), ':', i[1]

#____________________________________________________________
#____________________________________________________________
# 
def GetOptions(): 
    """Get options/optional parameter dictionary for fminCMA() and for the constructor
    of class CMAEvolutionStrategy, wrapper for calling fminCMA([],[],[])

      :Example: 
         >>> from cma import GetOptions, fminCMA, frosen
         >>> opts = GetOptions()
         >>> for k,v in sorted(opts.items()): print k.rjust(32), '=', repr(v)

                     CMAdiagonal = '0*100*N/sqrt(popsize)'
                           CMAon = 1
                       CMArankmu = 1
                  CMArankmualpha = 0.29999999999999999
                    evalparallel = False
                         ftarget = -1.#INF
                      incpopsize = 2
                       maxfevals = 1.#INF
                         maxiter = 'long(1e3*N**2/sqrt(popsize))'
                         popsize = '4+int(3*log(N))'
                        restarts = 0
                       tolfacupx = 1000.0
                          tolfun = 9.9999999999999998e-013
                      tolfunhist = 0
                            tolx = 9.9999999999999994e-012
                     verb_append = 0
                       verb_disp = 100
             verb_filenameprefix = 'outcmaes'
                        verb_log = 1

         >>> opts['CMAdiagonal'] = 0 
         >>> fminCMA(frosen, range(10), 1, **opts)

      :Note: options restarts and evalparallel are ignored for the class CMAEvolutionStrategy
    """
    return fminCMA([],[],[]) # assembles the keyword-arguments in a dictionary
    
#____________________________________________________________
#____________________________________________________________
# 
def fminCMA(func, x0, sigma0, args=()
            , tolfun = 1e-12  # prefered termination criterion
            , tolfunhist = 0 
            , tolx = 1e-11
            , tolfacupx = 1e3 # terminate on divergent behavior
            , ftarget = -inf  # target function value
            , maxiter = 'long(1e3*N**2/sqrt(popsize))'
            , maxfevals = inf
            , evalparallel = False
            , verb_disp = 100
            , verb_log = 1
            , verb_filenameprefix = 'outcmaes'
            , verb_append = 0 
            , popsize = '4+int(3*log(N))'
            , restarts = 0
            , incpopsize = 2
            , CMAon = True
            , CMAdiagonal = '0*100*N/sqrt(popsize)' # TODO 4/ccov_separable? 
            , CMAmu = None    # default is lambda/2
            , CMArankmu = True 
            , CMArankmualpha = 0.3
            , scaling_of_variables = None
            , typical_x = None
            ): 
    """fminCMA -- stochastic optimizer CMA-ES for non-convex function minimization

    :Calling Sequences:
        fminCMA([],[],[]) -- alias GetOptions(), returns all optional 
            parameters, that is all keyword arguments to fminCMA with 
            their default values in a dictionary.
        fminCMA(func, x0, sigma0) -- minimizes func starting
            at x0 with standard deviation sigma0 (step-size)
        fminCMA(func, x0, sigma0, ftarget=1e-5) -- minimizes func 
            up to target function value 1e-5
        fminCMA(func, x0, sigma0, args=('f',), **options) -- minimizes 
            func called with an additional argument 'f'. options
            is a dictionary with additional keyword arguments, e.g.
            delivered by GetOptions(). 

    :Arguments:
        func -- function to be minimized. Called as
            func(x,*args). x is a one-dimensional numpy.ndarray. func
            can return None (only if evalparallel argument is False),
            which is interpreted as outright rejection of solution x
            and invokes an immediate resampling and (re-)evaluation
            of a new solution not counting as function evaluation
        x0 -- initial canditate solution, initial guess of minimum
        sigma0 -- scalar, initial standard deviation in each coordinate.
            The variables in func should be scaled such that they
            presumably have similar sensitivity. See also option
            scaling_of_variables. 

    :Keyword Arguments:
        options -- keyword arguments in a dictionary
        args=() -- additional arguments to func
        tolfun=1e-12 -- termination criterion: tolerance in function value
        tolfunhist=0 -- termination criterion: tolerance in function value history
        tolx=1e-11 -- termination criterion: tolerance in x-changes
        tolfacupx=1e3 -- termination when step-size increases by tolfacupx. In 
            this case the initial step-size was chosen far too small or
            the objective function unintentionally indicates better
            solutions far away from the initial solution x0. 
        ftarget=-inf -- target function value, minimization
        maxiter='long(1e3*N**2/sqrt(popsize))' -- maximum number of iterations
        maxfevals=inf -- maximum number of function evaluations
        scaling_of_variables -- scale for each variable, sigma0 is interpreted
            w.r.t. this scale, in that internally the variable is divided by
            scaling_of_variables, default is ones(N). 
        evalparallel -- func is called with a list of solutions (only fminCMA())
        verb_disp=100 -- verbosity: console display every verb_disp iteration
        verb_log=1 -- verbosity: write data every verb_log iteration
        verb_filenameprefix='outcmaes' -- output filename prefix
        verb_append=0 -- append/overwrite old files
        popsize='4+int(3*log(N))' -- population size lambda
        restarts=0 -- number of restarts (only fminCMA())
        incpopsize=2 -- increment factor for popsize before each restart
        CMAon=1 -- 0 for no adaptation of the covariance matrix
        CMAdiagonal='0*100*N/sqrt(popsize)' -- number of iterations with diagonal
            covariance matrix, True for always. 
        CMArankmu=1 -- 0 for omitting rank-mu update of covariance matrix
        CMArankmualpha=0.3 -- factor of rank-mu update for mu=1

    :Return: res = (xopt, fopt, dict('mean', 'stopdict', 'recent', 'opts', 'es'))
        xopt -- best evaluated solution equals to res[2]['es'].Best()[0]
        fopt -- respective function value, equals to res[2]['es'].Best()[1]
        mean -- final mean solution point
        recent -- best solution point of the last iteration
        opts -- actually appied options
        es -- the class CMAEvolutionStrategy used

    :Notes:
        This function is an interface to the class CMAEvolutionStrategy. The
        class can be used when full control over the iteration loop of the
        optimizer is desired. For example, if a callback is needed, the
        (short) source code of this functions can be used as starting point
        (type 'pylab.source(cma.fminCMA)').

    :Example:
        import numpy, pylab, cma
        res = cma.fminCMA(cma.frosen, 0.1*numpy.ones(10), 0.5)
        print 'solution: ', res[0]
        dat = cma.PlotCMAdata()
        pylab.savefig('myfirstrun')
        pylab.show()  # to continue you might need to close the shown window 

    :See also: CMAEvolutionStrategy, PlotCMAdata(), GetOptions(), scipy.optimize() 
    """

    opts = locals()
    del opts['func']
    del opts['args']
    del opts['x0']      # is not optional
    del opts['sigma0']  # is not optional
    
    if not func:  # return available options in a dictionary
        return opts

    irun = 0
    tic = time.time()
    while 1 :
        es = CMAEvolutionStrategy(x0, sigma0, opts) 
        if irun > 0:
            es.updateBest(x_prev, f_prev, ev_prev-es.countevals)
        while not es.Stop():
            X = es.Ask()  # get list of new solutions
            if opts['evalparallel']:
                fit = func(X, *args)  # without copy
            else:
                fit = len(X) * [None]
                for k in xrange(len(X)) :
                    while fit[k] == None:
                        fit[k] = func(X[k].copy(), *args)  # call func on solutions
                        if fit[k] == None:                 # resample None fitness
                            X[k] = es.Ask(1)[0]            
                        
            es.Tell(X, fit)  # prepare for next iteration

            # console display 
            if 0.1*(es.countiter-1) % verb_disp == 0:
                # Iterat, #Fevals:   Function Value    (median,worst) |Axis Ratio|idx:Min SD idx:Max SD time
                print 'Iterat #Fevals   function value      axis ratio  sigma   time [m:s]'
                sys.stdout.flush()
            if verb_disp and (es.Stop() or es.countiter < 4 
                              or es.countiter % verb_disp == 0): 
                print repr(es.countiter).rjust(5), repr(es.countevals).rjust(7), \
                      '%.16e' % (min(fit)), \
                      '%4.1e' % (es.D.max()/es.D.min()), \
                      '%6.2e' % es.sigma, 
                toc = time.time() - tic
                print str(int(toc/60))+':'+str(round(toc%60,1))
                if es.countiter < 4:
                    sys.stdout.flush()

        # end while not es.stop()

        if opts['evalparallel']:
            es.updateBest(es.mean, func((es.mean.copy(),), *args), 1)
        else:
            es.updateBest(es.mean, func(es.mean.copy(), *args), 1)
        es.countevals += 1

        # final message
        for k,v in es.stopdict.iteritems():
            print k, ':', v

        irun += 1
        if irun > restarts or 'ftarget' in es.stopdict or 'maxfunevals' in es.stopdict:
            break
        x_prev = es.out['best_x']
        f_prev = es.out['best_f']
        ev_prev = es.out['best_evals']
        opts['verb_append'] = es.countevals
        opts['popsize'] = incpopsize * es.sp.popsize # TODO: use rather options? 

    # while irun

    return (es.out['best_x'].copy(), es.out['best_f'], 
            dict((('mean', es.mean.copy())
                  ,('stopdict', es.stopdict.copy()) 
                  ,('recent', es.out['recent_x'].copy()) 
                  ,('out', es.out)
                  ,('opts', es.opts.copy())
                  ,('cma', es)
                  )))
                  # TODO refine output, can #args be flexible?
                  # is this well usable as it is now?
# end fminCMA()

#____________________________________________________________
#____________________________________________________________
def _evalOption(o, default, loc=None):
    """Evaluates input o as option in environment loc
    """
    if o == str(o):
        val = eval(o, globals(), loc)
    else:
        val = o
    if val in (None, (), [], ''):  # TODO: {} in the list gives an error
        val = eval(str(default), globals(), loc)
    return val
    
#____________________________________________________________
#____________________________________________________________
# 
class CMAEvolutionStrategy(object):
    """CMA-ES stochastic optimizer class with ask-and-tell interface
    
       :Example:
            import sys, cma
            from numpy.random import rand

            options = {}
            options['verb_log'] = 0  # no output files written
            es = cma.CMAEvolutionStrategy(rand(9), 0.5, options)  # a new instance

            while not es.stop:
                X = es.Ask()  # get list of new solutions
                fit = []
                for x in X:
                    fit.append(cma.frosen(x.copy()))  # call func, copy() is superfluous for frosen
                es.Tell(X, fit)  # besides for termination only the ranking in fit is used
                # display some output
                if es.countiter < 3 or es.countiter % 100 == 0 or es.stop:
                    print es.countiter, es.countevals, es.fit.fit[0]
                    sys.stdout.flush()

            cma.pprint(es.out)

            Look at fminCMA() to find another example of how to use this class
            (type 'pylab.source(cma.fminCMA)').

       :See also: fminCMA()
            
    """

    def _getmean(self):
        """mean value
        """
        return self.mean

    def _setmean(self, m):
        """mean value setter
        """
        self.mean = m
        
    #____________________________________________________________
    #____________________________________________________________
    def Stop(self):
        """Return termination status, either True of False. 
        See also StopList(), StopDict() 
        """

        return self.StopList() != []

    stop = property(Stop)    # getter function is Stop()
    mean_x = property(_getmean, _setmean) # not really in use? 

    #____________________________________________________________
    #____________________________________________________________
      
    def _addStop(self, key, cond, val=None):
        if cond:
            if key in self.opts.keys():
                item = self.opts[key]
            else:
                item = val
            self.stoplist.append(key)
            self.stopdict[key] = item
    
    #____________________________________________________________
    #____________________________________________________________
    def _testStop(self):
        """Test termination criteria and update respective lists
        see also StopList() and StopDict(). 
        """
    #____________________________________________________________

        if self.countiter == self.stoplastiter:
            if self.countiter == 0:
                self.stoplist = []
                self.stopdict = {}
            return 

        self.stoplastiter = self.countiter

        self.stoplist = []
        self.stopdict = {}
        
        N = self.N
        opts = self.opts
        
        # fitness: generic criterion, user defined w/o default
        self._addStop('ftarget',
                     self.out['best_f'] < opts['ftarget'])
        # maxiter, maxfevals: generic criteria
        self._addStop('maxfevals',
                     self.countevals - 1 >= opts['maxfevals'])
        self._addStop('maxiter',
                     self.countiter >= opts['maxiter'])
        # tolx, tolfacupx : generic criteria
        # tolfun, tolfunhist (CEC:tolfun includes hist) 
        self._addStop('tolx',
                     all([self.sigma*xi < opts['tolx'] for xi in self.pc]) and \
                     all([self.sigma*xi < opts['tolx'] for xi in sqrt(self.dC)]))
        self._addStop('tolfacupx',
                     any([self.sigma * sig > self.inargs['sigma0']*opts['tolfacupx']
                          for sig in sqrt(self.dC)]))
        self._addStop('tolfun',
                     self.fit.fit[-1] - self.fit.fit[0] < opts['tolfun'] and \
                     max(self.fit.hist) - min(self.fit.hist) < opts['tolfun'])
        self._addStop('tolfunhist',
                     len(self.fit.hist) > 9 and \
                     max(self.fit.hist) - min(self.fit.hist) <  opts['tolfunhist'])

        # TODO: add option stoponequalfunvals and test here...

        # non-user defined, method specific
        # noeffectaxis (CEC: 0.1sigma), noeffectcoord (CEC:0.2sigma), conditioncov
        self._addStop('noeffectcoord',
                     any([self.mean[i] == self.mean[i] + 0.2*self.sigma*sqrt(self.dC[i])
                          for i in xrange(N)]))
        if opts['CMAdiagonal'] is not True and self.countiter > opts['CMAdiagonal']: 
            self._addStop('noeffectaxis',
                         all([self.mean[i] == self.mean[i] +
                              0.1*self.sigma * self.D[self.countiter%N]
                              * self.B[i,self.countiter%N] for i in xrange(N)]))
        self._addStop('conditioncov',
                     self.D[-1] > 1e7 * self.D[0], 1e14)  # TODO 
        return self.StopList() != []

    #____________________________________________________________
    #____________________________________________________________
    def StopList(self) :
        """Get list of satisfied termination criteria, empty if none
        """
        self._testStop()
        return self.stoplist[:] # returns a copy, such that it is not modified later

    #____________________________________________________________
    #____________________________________________________________
    def StopDict(self) :
        """Get a dictionary of satisfied termination criteria, empty if none
        """
        self._testStop()
        return self.stopdict.copy() 

    #____________________________________________________________
    #____________________________________________________________
    def Best(self) :
        """Return (xbest, fbest) -- best solution so far evaluated together
              with its function value
        """

        return (self.out['best_x'], self.out['best_f']) 

    #____________________________________________________________
    #____________________________________________________________
    def Ask(self, number=None) :
        """Get new candidate solutions, sample from a multi-variate
           normal distribution. 
        :Parameters:
            number -- number of returned solutions, by default the
            population size popsize/lambda. 
        :Return:
            list of N-dimensional candidate solutions to be evaluated
        """
    #____________________________________________________________

        flgseparable = self.opts['CMAdiagonal'] is True \
                       or self.countiter <= self.opts['CMAdiagonal'] 
        if number == None or number < 1:
            number = self.sp.popsize

        # update parameters for sampling the distribution
        if 1 < 3 and self.sp.CMAon and \
               self.countiter > self.updateeigen + 1./(self.sp.c1+self.sp.cmu)/self.N/10.:
            # updateeigen is always up-to-date in the diagonal case 
            self.C = 0.5 * (self.C + self.C.T)
            # self.C = numpy.triu(self.C) + numpy.triu(self.C,1).T  # should work as well
            # self.D, self.B = eigh(self.C) # hermitian, ie symmetric C is assumed
            if 1 < 3:  # is at least 2 times faster in 100-D(!) and easy to install 
                         # and eigenvectors are orthogonal 
                import pygsl.eigen
                self.D, self.B = pygsl.eigen.eigenvectors(self.C) 
                idx = numpy.argsort(self.D)
                self.D = self.D[idx]
                self.B = self.B[:,idx] # self.B[i] is the i+1-th row and not an eigenvector
                
            elif 11 < 3: # crashes in 200-D
                self.D, self.B = numpy.linalg.eig(self.C) # self.B[i] is a row and not an eigenvector
                idx = numpy.argsort(self.D)
                self.D = self.D[idx]
                self.B = self.B[:,idx]
            elif 1 < 3:  # is overall two;ten times slower in 10;20-D
                import eigen  # TODO: profile eigen code, move eigen code in this file here,
                              #   also consider to do a third version linking the C-code
                              #   looking at the performance
                self.D, self.B = eigen.eig(self.C)
                idx = numpy.argsort(self.D)
                self.D = self.D[idx]
                self.B = self.B[:,idx]
                # assert(sum(self.D-DD) < 1e-6)
                # assert(sum(sum(dot(BB, BB.T)-numpy.eye(self.N))) < 1e-6)
                # assert(sum(sum(dot(BB * DD, BB.T) - self.C)) < 1e-6)
            else:
                pass
            
            # assert(all(self.B[self.countiter % self.N] == self.B[self.countiter % self.N,:]))
                
            # EVecs with same EVals are not orthogonal! 
            if 11 < 3 and any(abs(sum(self.B[:,0:self.N-1] * self.B[:,1:], 0)) > 1e-6): # TODO incomment 
                print 'B is not orthogonal'
                print self.D
                print sum(self.B[:,0:self.N-1] * self.B[:,1:], 0) 
            else:
                # TODO remove as it is O(N^3)
                # assert(sum(abs(self.C - dot(self.D * self.B,  self.B.T))) < N**2*1e-11) 
                pass
            self.D **= 0.5
            self.updateeigen = self.countiter

        # sample distribution
        self.arx = [] 
        self.ary = [] 
        for k in xrange(number):
            if 1 < 2 or self.N > 40:
                self.ary.append(dot(self.B, self.D*numpy.random.randn(self.N)))
            else:
                if self.countiter == 0 and k == 0:
                    print 'sobol numbers applied'
                # sobol sequence: does strongly bias sigma if mu > 1
                self.ary.append(dot(self.B, self.D*sobolrandn(self.N))) 
            # self.arx.append(self.mean + self.sigma * self.ary[k])
            self.arx.append(self.gp.pheno(self.mean + self.sigma * self.ary[k]))

        # make geno-pheno transformation here
        return self.arx  # this is just a reference and not so terribly useful to keep 

    #____________________________________________________________
    #____________________________________________________________
    def _computeParameters(self, N, popsize, opts):
        """Compute strategy parameters mainly depending on
        population size """
        #____________________________________________________________
        # learning rates cone and cmu as a function
        # of the degrees of freedom df
        def cone(df, mucov, N):
            return 1. / (df + 2.*sqrt(df) + mucov/N)

        def cmu(df, mucov, alphamu):
            return (alphamu + mucov - 2. + 1./mucov) / (df + 4.*sqrt(df) + mucov/2.)

        #____________________________________________________________
        sp = _Struct()  # just a hack 
        sp.popsize = popsize
        sp.mu_f = sp.popsize / 2.0  # float value of mu
        if opts['CMAmu'] != None: 
            sp.mu_f = opts['CMAmu']
            print 'mu =', sp.mu_f
        sp.mu = int(sp.mu_f + 0.4999) # round down for x.5
        sp.weights = log(sp.mu_f+0.5) - log(1+arange(sp.mu))
        sp.weights /= sum(sp.weights)
        sp.mueff = 1./sum(sp.weights**2)
        sp.cs = (sp.mueff+2)/(N+sp.mueff+3)
        sp.cc = 4. / (N+4.)
        sp.cc_sep = sp.cc  # 1. / sqrt(N)  # TODO: should this stay? # 2. / (sqrt(N)+1.) 
        sp.mucov = sp.mueff
        sp.rankmualpha = _evalOption(opts['CMArankmualpha'], 0.3)
        if -1 > 0:
            sp.mucov = 1
            print 'mucov=1'
        sp.c1 = cone((N**2+N)/2, sp.mucov,N) # 2. / ((N+1.3)**2 + sp.mucov)
        sp.c1_sep = cone(N, sp.mucov, N) # 2. / ((N+1.3)**2 + sp.mucov)
        if -1 > 0:
            sp.c1 = 0.
            print 'c1 is zero'
        if opts['CMArankmu'] != 0:  # also empty
            sp.cmu = min(1-sp.c1, cmu((N**2+N)/2, sp.mucov, sp.rankmualpha))
            sp.cmu_sep = min(1-sp.c1_sep, cmu(N, sp.mucov, sp.rankmualpha))
        else:
            sp.cmu = sp.cmu_sep = 0

        sp.CMAon = sp.c1 + sp.cmu > 0  # mainly for historical reasons
        # print sp.c1_sep / sp.cc_sep

        if not opts['CMAon'] and opts['CMAon'] != None and opts['CMAon'] != []: 
            sp.CMAon = False
            # sp.c1 = sp.cmu = sp.c1_sep = sp.cmu_sep = 0
            print 'covariance matrix adaptation turned off'
        sp.damps = (1 + 2*max(0,sqrt((sp.mueff-1)/(N+1))-1)) + sp.cs 
        if -1 > 0:
            sp.damps = 30 * sp.damps  # 1e99 # (1 + 2*max(0,sqrt((sp.mueff-1)/(N+1))-1)) + sp.cs; 
            print 'damps is', sp.damps

        return sp  # the only existing reference to sp is passed here

    #____________________________________________________________
    #____________________________________________________________
    #____________________________________________________________
    #____________________________________________________________
    def __init__(self, x0, sigma0, inopts = {}):
        """
        :Parameters:
            x0 -- initial solution, starting point
            sigma0 -- initial standard deviation.  The problem
                variables should have been scaled, such that a single
                standard deviation on all variables is useful and the
                optimum is expected to lie within x0 +- 3*sigma0.
            inopts -- options, a dictionary according to the parameter list
              of function fminCMA(), see there and GetOptions()
        """

        #____________________________________________________________
        self.inargs = locals().copy()
        del self.inargs['self'] # otherwise the instance self has a cyclic reference
        self.inopts = inopts
        defopts = fminCMA([],[],[])
        opts = defopts.copy()
        if inopts: # filling of opts with inopts while checking for unknow keys in inopts
            if not isinstance(inopts, dict): 
                raise Error('options argument must be a dict')
            for key in inopts.keys():
                if not defopts.has_key(key):
                    raise Error(key + ' is invalid key entry for options argument, see GetOptions()')
                opts[key] = inopts[key]

        if x0 == str(x0) :
            self.mean = array(eval(x0), copy=True)
        else :
            self.mean = array(x0, copy=True) # does not have column or row, is just 1-D

        if self.mean.ndim != 1:
            raise Error('x0 must be 1-D array')
        
        self.N = self.mean.shape[0]
        N = self.N
        self.mean.resize(N) # 1-D array
        self.sigma = sigma0
        popsize = _evalOption(opts['popsize'], defopts['popsize'], locals())

        # extract/expand options 
        for key in opts.keys():
            if key.find('filename') < 0:
                opts[key] = _evalOption(opts[key], defopts[key], locals())
            elif not opts[key]:
                opts[key] = defopts[key]
        
        self.opts = opts

        self.gp = _GenoPheno(opts['scaling_of_variables'], opts['typical_x'])  # qqq
        self.mean = self.gp.geno(self.mean)
        self.sp = self._computeParameters(N, popsize, opts)
        self.sp0 = self.sp

        # initialization of state variables
        self.countiter = 0
        self.countevals = 0
        self.ps = numpy.zeros(N)
        self.pc = numpy.zeros(N)

        if self.opts['CMAdiagonal'] > 0 :  # linear time and space complexity ((True>0)==True)
            self.B = array(1) # works fine with dot(self.B, anything) and self.B.T
            self.C = numpy.ones(N)
        else :
            self.B = numpy.eye(N) # identity(N), do not from matlib import *, as eye is a matrix there
            # self.C = numpy.diag(1 + 0.01*numpy.random.rand(N)) # prevent equal eigenvals, a hack for numpy.linalg TODO
            self.C = numpy.eye(N)
        self.D = numpy.ones(N)
        self.dC = numpy.ones(N)
        self.updateeigen = self.countiter

        out = {}
        # out.best = _Struct()
        out['best_f'] = inf
        out['best_x'] = []
        out['best_evals'] = 0
        out['opts_in'] = inopts.copy()
        out['hsigcount'] = 0
        out['hsiglist'] = [] # TODO: remove at least from visible output at some point
        self.out = out
        
        self.const = _Struct()
        self.const.chiN = N**0.5*(1-1./(4.*N)+1./(21.*N**2)) # expectation of 
				      #   ||N(0,I)|| == norm(randn(N,1))
                                      # normalize recombination weights array

        # attribute for stopping criteria in function stop 
        self.stoplastiter = 0 
        self.fit = _Struct()
        self.fit.fit = []   # not really necessary
        self.fit.hist = []

        if self.opts['verb_log'] > 0 and not self.opts['verb_append']:
            self.WriteHeaders()
        if self.opts['verb_append'] > 0:
            self.countevals = self.opts['verb_append']

        # say hello
        if opts['verb_disp'] > 0:
            #if self.sp.mu == 1:
            #    print '(%d,%d)-CMA-ES' % (self.sp.mu, self.sp.popsize),
            #else:
            #    print '(%d_w,%d)-CMA-ES' % (self.sp.mu, self.sp.popsize), 
            #print '(mu_w=%2.1f,w_1=%d%%)' % (self.sp.mueff, int(100*self.sp.weights[0])),
            #print 'in dimension %d ' % N # + func.__name__ 
            if opts['CMAdiagonal'] and self.sp.CMAon:
                s = ''
                if opts['CMAdiagonal'] is not True:
                    s = ' for '
                    if opts['CMAdiagonal'] < Inf:
                        s += str(int(opts['CMAdiagonal']))
                    else:
                        s += str(floor(opts['CMAdiagonal']))
                    s += ' iterations'
                    s += ' (1/ccov=' + str(round(1./(self.sp.c1+self.sp.cmu))) + ')'
                print '   Covariance matrix is diagonal' + s 

    #____________________________________________________________
    #____________________________________________________________
    def Tell(self, points, function_values) :  
        """Pass objective function values to CMA-ES to prepare for next iteration

        :Arguments:
           points -- list or array of candidate solution points,
              most presumably before delivered by method Ask().
           function_values -- list or array of objective function values 
              associated to the respective points. Beside termination 
              decisions, only the ranking of values in function_values
              is used. 
        """
    #------------------------------------------------------------
        # from numpy import sum, outer, dot

        lam = len(points)
        pop = self.gp.geno(points)
        if lam != array(function_values).shape[0] :
            raise Error('for each candidate solution '
                        + 'a function value must be provided')
        if lam < 3: 
            raise Error('population ' + str(lam) + ' is too small')
        N = self.N
        if lam != self.sp.popsize:
            # print 'population size changed'
            self.sp = self._computeParameters(N, lam, self.opts) 
        sp = self.sp

        self.countiter += 1  # >= 1 now
        self.countevals += sp.popsize
        flgseparable = self.opts['CMAdiagonal'] is True \
                       or self.countiter <= self.opts['CMAdiagonal'] 
        if not flgseparable and len(self.C.shape) == 1:  # C was diagonal ie 1-D
            self.B = numpy.eye(N) # identity(N)
            self.C = numpy.diag(self.C)
            idx = numpy.argsort(self.D)
            self.D = self.D[idx]
            self.B = self.B[:,idx]

        fit = self.fit  # make short cut
        fit.idx = numpy.argsort(function_values)
        fit.fit = array(function_values)[fit.idx]

        if 11 < 3: # qqq mirrored vectors need special treatment
            # need to know whether 0,1 etc. are mirrors or 1,2 etc
            # but how? Maybe better in fminCMA ? 
            pass
        
        fit.hist.insert(0, fit.fit[0])
        if len(fit.hist) > 10 + 30*N/sp.popsize:
            fit.hist.pop()

        # compute new mean and sort pop
        mold = self.mean
        del self.mean  # not sure why, can be removed? 
        pop = array(pop)[fit.idx] # only arrays can be multiple indexed
        self.mean = sum(sp.weights * array(pop[0:sp.mu]).T, 1) 
        
        # evolution paths
        self.ps = (1-sp.cs) * self.ps + \
                  (sqrt(sp.cs*(2-sp.cs)*sp.mueff)  / self.sigma) \
                  * dot(self.B, (1./self.D) * dot(self.B.T, (self.mean - mold)))

        hsig = (sqrt(sum(self.ps**2)) / sqrt(1-(1-sp.cs)**(2*self.countiter)) 
                / self.const.chiN) < 1.4 + 2./(N+1)
        if -1 > 0:
            hsig = 1
        self.out['hsigcount'] += 1 - hsig
        if not hsig:
            self.out['hsiglist'].append(self.countiter)
        
        if -1 > 0:
            if not hsig:  
                print self.countiter, ': hsig-stall'
        if -1 > 0:
            hsig = 1 # TODO: in 4-D and damps=1e99 hsig leads to premature convergence of C, put correction term
            if self.countiter == 1:
                print 'hsig=1'
        
        cc = sp.cc
        if flgseparable:
            cc = sp.cc_sep
        
        self.pc = (1-cc) * self.pc + \
                  hsig * sqrt(cc*(2-cc)*sp.mueff) \
                  * (self.mean - mold) / self.sigma

        # covariance matrix adaptation 
        if sp.CMAon:
            assert sp.c1 + sp.cmu < sp.mueff / N
            # default quadratic case 
            if not flgseparable:
                Z = (pop[0:sp.mu]-mold)/self.sigma
                Z = dot(sp.weights*Z.T,Z)
                if -1 > 0:
                    Z = numpy.zeros((N,N))
                    for k in xrange(sp.mu): # 3 to 5 times slower
                        z = (pop[k]-mold) 
                        Z += outer((sp.weights[k]/ self.sigma**2) * z, z)
                self.C = (1-sp.c1-sp.cmu) * self.C + \
                         outer(sp.c1 * self.pc, self.pc) + \
                         sp.cmu * Z # TODO? multiplying sp.cmu can be done before with mu instead of N^2 multiplications
                self.dC = numpy.diag(self.C)

            else : # separable/diagonal linear case 
                c1, cmu = sp.c1_sep, sp.cmu_sep 
                assert(c1+cmu <= 1)
                Z = numpy.zeros(N)
                for k in xrange(sp.mu):
                    z = (pop[k]-mold) / self.sigma  # TODO see above
                    Z += sp.weights[k] * z * z  # is 1-D
                self.C = (1-c1-cmu) * self.C + c1 * self.pc * self.pc + cmu * Z
                self.dC = self.C
                self.D = sqrt(self.C)  # C is a 1-D array
                self.updateeigen = self.countiter
                
        # step-size adaptation
        self.sigma *= numpy.exp((sp.cs/sp.damps) * \
                                (sqrt(sum(self.ps**2))/self.const.chiN - 1))

        # TODO increase sigma in case of a plateau? 

        # update output data
        self.updateBest(pop[0], fit.fit[0], fit.idx[0]+1)
        out = self.out
        out['opts'] = self.opts    # some options might become dynamic in future
        out['recent_x'] = pop[0]
        out['recent_f'] = fit.fit[0]
        out['funevals'] = self.countevals
        out['iterations'] = self.countiter
        out['mean'] = self.mean
        
        # output
        if self.opts['verb_log'] > 0 and (self.countiter < 4 or
                                          self.countiter % self.opts['verb_log'] == 0):
            self.WriteOutput(self.gp.pheno(pop[0]))

    # END def tell

    #____________________________________________________________
    #____________________________________________________________
    def updateBest(self, x, fun_value, evals=None):
        """Update best ever visited solution

        :Arguments:
           x -- solution point
           fun_value -- respective objective function value
           evals -- function evaluation number when x was evaluated

        :Description:
           When fun_value is smaller than the function value of the
           so far stored best solution, the best solution is replaced. 
        """
    #------------------------------------------------------------
        if evals == None:
            evals = self.sp.popsize+1
        
        if self.out['best_f'] > fun_value: 
            self.out['best_x'] = x.copy() 
            self.out['best_f'] = fun_value
            self.out['best_evals'] = self.countevals - self.sp.popsize + evals

    #____________________________________________________________
    #____________________________________________________________
    def WriteOutput(self, xrecent=[]):
        """
        write output to files
         :Arguments:
           xrecent -- recent best x-vector (candidate solution, phenotype)
        """
    #------------------------------------------------------------
        flgseparable = self.opts['CMAdiagonal'] is True \
                       or self.countiter <= self.opts['CMAdiagonal'] 
        if 1 < 3:
            # fit
            fn = self.opts['verb_filenameprefix'] + 'fit.dat'
            try : 
                f = open(fn, 'a')
                f.write(str(self.countiter) + ' ' 
                        + str(self.countevals) + ' '
                        + str(self.sigma) + ' '
                        + str(self.D.max()/self.D.min()) + ' '
                        + str(self.out['best_f']) + ' '
                        + str(self.fit.fit[0]) + ' '
                        + str(self.fit.fit[int(self.sp.popsize/2)]) + ' '
                        + str(self.fit.fit[-1]) + ' '
                        + '\n')
                f.close()
            except :
                if self.countiter == 1 :
                    print 'could not open/write file', fn
            # axlen
            fn = self.opts['verb_filenameprefix'] + 'axlen.dat'
            try : 
                f = open(fn, 'a')
                diagout = self.D
                if flgseparable:
                    diagout = numpy.sort(self.D)
                f.write(str(self.countiter) + ' ' 
                        + str(self.countevals) + ' '
                        + str(self.sigma) + ' '
                        + str(self.D.max()) + ' '
                        + str(self.D.min()) + ' '
                        + string.join(map(str,diagout))
                        + '\n')
                f.close()
            except :
                if self.countiter == 1 :
                    print 'could not open/write file', fn
            # stddev
            fn = self.opts['verb_filenameprefix'] + 'stddev.dat'
            try : 
                f = open(fn, 'a')
                f.write(str(self.countiter) + ' ' 
                        + str(self.countevals) + ' '
                        + str(self.sigma) + ' '
                        + '0 0 '
                        + string.join(map(str,self.sigma*sqrt(self.dC)))
                        + '\n')
                f.close()
            except :
                if self.countiter == 1 :
                    print 'could not open/write file', fn
            fn = self.opts['verb_filenameprefix'] + 'xmean.dat'
            try : 
                f = open(fn, 'a')
                f.write(str(self.countiter) + ' ' 
                        + str(self.countevals) + ' '
                        # + str(self.sigma) + ' '
                        + '0 0 0 '
                        + string.join(map(str,self.mean))
                        + '\n')
                f.close()
            except :
                if self.countiter == 1 :
                    print 'could not open/write file', fn
            fn = self.opts['verb_filenameprefix'] + 'xrecentbest.dat'
            try : 
                f = open(fn, 'a')
                f.write(str(self.countiter) + ' ' 
                        + str(self.countevals) + ' '
                        + str(self.sigma) + ' '
                        + '0 '
                        + str(self.fit.fit[0]) + ' '
                        + string.join(map(str,xrecent))
                        + '\n')
                f.close()
            except :
                if self.countiter == 1 :
                    print 'could not open/write file', fn

    #____________________________________________________________
    #____________________________________________________________
    def WriteHeaders(self):
        """
        write headers to files, overwrites existing files
        """
    #------------------------------------------------------------
        if self.opts['verb_log'] == 0:
            return

        # write headers for output
        fn = self.opts['verb_filenameprefix'] + 'fit.dat'
        try : 
            f = open(fn, 'w');
            f.write('% # columns="iteration, evaluation, sigma, axis ratio, ' +
                    'bestever, best, median, worst objective function value, ' +
                    'further objective values of best", ' +
                    time.asctime() + 
                    # strftime("%Y/%m/%d %H:%M:%S", localtime()) + # just asctime() would do
                    '\n')
            f.close()
        except :
            print 'could not open file', fn
        fn = self.opts['verb_filenameprefix'] + 'axlen.dat'
        try : 
            f = open(fn, 'w');
            f.write('%  columns="iteration, evaluation, sigma, max axis length, ' +
                    ' min axis length, all principle axes lengths ' +
                    ' (sorted square roots of eigenvalues of C)", ' +
                    time.asctime() + 
                    '\n')
            f.close()
        except :
            print 'could not open file', fn
        fn = self.opts['verb_filenameprefix'] + 'stddev.dat'
        try : 
            f = open(fn, 'w');
            f.write('% # columns=["iteration, evaluation, sigma, void, void, ' +
                    ' stds==sigma*sqrt(diag(C))", ' +
                    time.asctime() + 
                    '\n')
            f.close()
        except :
            print 'could not open file', fn
        fn = self.opts['verb_filenameprefix'] + 'xmean.dat'
        try : 
            f = open(fn, 'w');
            f.write('% # columns="iteration, evaluation, void, void, void, xmean", ' +
                    '\n')
            f.close()
        except :
            print 'could not open file', fn
        fn = self.opts['verb_filenameprefix'] + 'xrecentbest.dat'
        try : 
            f = open(fn, 'w');
            f.write('% # iter+eval+sigma+0+fitness+xbest ' +  # TODO find out how to get random state
                    time.asctime() + 
                    '\n')
            f.close()
        except :
            print 'could not open file', fn

    # end def WriteHeaders

# end class cma 

#____________________________________________________________
#____________________________________________________________
# 
def downsampling(filename='outcmaes', factor=10):
    """
    rude downsampling of CMA-ES data file by factor keeping the
    first factor entries
    :Parameters:
       filename: prefix of files used
       factor: downsampling factor
    :Output: filename+'down' files
    """
    for name in ('axlen','fit','stddev', 'xmean','xrecentbest'):
        f = open(filename+'down'+name+'.dat','w')
        iline = 0
        for line in open(filename+name+'.dat'):
            if iline < factor or iline % factor == 0:
                f.write(line)
            iline += 1
        f.close()
        
#____________________________________________________________
#____________________________________________________________
# 

from pylab import ioff, ion, draw, show, isinteractive 
def PlotCMAdata(in1=[], in2=1) :
    """
    plots data from files written by method Tell() in method WriteOutput()
    :Parameters:
       in1 -- filename or figure number
       in2 -- 0==plot versus iteration count,
              1==plot versus funcion evaluation number

    :Example:
       dat = cma.PlotCMAdata() # the optimization might be still 
                               # running from a different shell
       pylab.show()
       pylab.savefig('fig325.png')
       pylab.close()
    """
    # pylab: prodedural interface for matplotlib
    from pylab import figure, subplot, semilogy, hold, plot, grid, ioff, ion, \
         axis, title, text, xlabel, draw, show, isinteractive 

    filenameprefix='outcmaes'
    abscissa = 1
    if in1 == str(in1):
        filenameprefix = in1;
    elif in1:
        figure(in1)
    else:
        figure(324)
    if in2 in (0,1):
        iabscissa = in2
    interactive_status = isinteractive()
    ioff() # prevents immediate drawing

    objectvarname = 'xmean'  # 'xrecentbest'

    if iabscissa == 0:
        xlab = 'iterations' 
    elif iabscissa == 1:
        xlab = 'function evaluations' 

    dat = _Struct()
    dat.x = array(_fileToMatrix(filenameprefix + objectvarname + '.dat')) 
    dat.f = array(_fileToMatrix(filenameprefix + 'fit.dat'))
    dat.D = array(_fileToMatrix(filenameprefix + 'axlen' + '.dat'))
    dat.std = array(_fileToMatrix(filenameprefix + 'stddev' + '.dat'))

    if dat.x.shape[1] < 100:
        minxend = int(1.05*dat.x[-1, iabscissa])
    else:
        minxend = 0

    foffset = 1e-49;
    dfit = dat.f[:,5]-min(dat.f[:,5]) 

    # dfit(dfit<1e-98) = NaN;

    ioff() # turns update off
    subplot(2,2,1)

    # TODO to be tested (larger indices): additional fitness data, for example constraints values
    if dat.f.shape[1] > 8:
        dd = abs(dat.f[:,8:]) + 10*foffset
        dd = numpy.where(dat.f[:,8:]==0, dd, None) # cannot be 
        semilogy(dat.f[:,iabscissa], dd, '-m')
        semilogy(dat.f[:,iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
        
    idx = numpy.where(dat.f[:,5]>1e-98)[0]  # positive values
    subplot(2,2,1)
    hold(False)
    semilogy(dat.f[idx,iabscissa], dat.f[idx,5]+foffset, '.b')
    hold(True)
    grid(True)
    
    idx = numpy.where(dat.f[:,5] < -1e-98)  # negative values
    subplot(2,2,1)
    semilogy(dat.f[idx, iabscissa], abs(dat.f[idx,5])+foffset,'.r')

    semilogy(dat.f[:,iabscissa],abs(dat.f[:,5])+foffset,'-b')
    semilogy(dat.f[:,iabscissa],dfit,'-c')
    semilogy(dat.f[-1,iabscissa]*numpy.ones(2), dat.f[-1,4]*numpy.ones(2), 'rd') 
    semilogy(dat.D[:,iabscissa], dat.D[:,-1] / dat.D[:,5], '-r') # AR
    semilogy(dat.std[:,iabscissa], dat.std[:,2],'-g'); # sigma
    ax = array(axis())
    ax[1] = max(minxend, ax[1])
    axis(ax)
    text(ax[0], ax[2], # 10**(log10(ax[2])+0.05*(log10(ax[3])-log10(ax[2]))),
         '.f_recent=' + repr(dat.f[-1,5]) )

    title('abs(f) (blue), f-min(f) (cyan), Sigma (green), Axis Ratio (red)')
    
    subplot(2,2,2)
    hold(False)
    plot(dat.x[:,iabscissa], dat.x[:,5:],'-')
    hold(True)
    grid(True)
    ax = array(axis())
    ax[1] = max(minxend, ax[1]) 
    axis(ax)
    ax[1] -= 1e-6 
    if dat.x.shape[1] < 100:
        yy = numpy.linspace(ax[2]+1e-6, ax[3]-1e-6, dat.x.shape[1]-5)
        yyl = numpy.sort(dat.x[-1,5:])
        idx = numpy.argsort(dat.x[-1,5:])
        idx2 = numpy.argsort(idx)
        plot([dat.x[-1,iabscissa], ax[1]], [dat.x[-1,5:], yy[idx2]], 'k-') # line from last data point
        # plot(array([dat.x[-1,iabscissa], ax[1]]),
        #      reshape(array([dat.x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
        plot(dot(dat.x[-1,iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-')
        for i in range(len(idx)):
            text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat.x[-1,5+idx[i]]))
    title('Object Variables (' + str(dat.x.shape[1]-5) + '-D)')

    subplot(2,2,3)
    hold(False)
    semilogy(dat.D[:,iabscissa], dat.D[:,5:], '-')
    hold(True)
    grid(True)
    ax = array(axis()) 
    ax[1] = max(minxend, ax[1]) 
    axis(ax) 
    title('Scaling (All Main Axes)') 
    xlabel(xlab) 

    subplot(2,2,4)
    hold(False)
    semilogy(dat.std[:,iabscissa], numpy.vstack([map(max,dat.std[:,5:]), map(min,dat.std[:,5:])]).T,
             '-m', linewidth=2)
    hold(True)
    grid(True)
    # remove sigma from stds
    dat.std[:,5:] = numpy.transpose(dat.std[:,5:].T / dat.std[:,2].T) 
    semilogy(dat.std[:,iabscissa], dat.std[:,5:], '-') 
    ax = array(axis())
    ax[1] = max(minxend, ax[1]) 
    axis(ax)
    if dat.std.shape[1] < 100:
        yy = numpy.logspace(numpy.log10(ax[2]), numpy.log10(ax[3]), dat.std.shape[1]-5)
        yyl = numpy.sort(dat.std[-1,5:])
        idx = numpy.argsort(dat.std[-1,5:])
        idx2 = numpy.argsort(idx)
        plot(dot(dat.std[-1,iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-') # vertical separator
        plot([dat.std[-1,iabscissa], ax[1]], [dat.std[-1,5:], yy[idx2]], 'k-') # line from last data point
        for i in xrange(len(idx)):
            text(ax[1], yy[i], ' '+str(idx[i]))
    title('Standard Deviations in All Coordinates')
    xlabel(xlab)
    draw()  # does not suffice
    if interactive_status:
        ion()  # turns interactive mode on (again)
        draw()  
    # show() # turns interactive mode on
    # ioff() # turns interactive mode off
    
    # TODO: does not bring up the figure, show() is needed, but enters interactive mode

    # semilogy(dat.f[:,iabscissa], dat.f[:,6])
    return dat

#____________________________________________________________

def _fileToMatrix(file_name):
#     try: 
        file = open(file_name, 'r')
        file.readline() # rudimentary, assume one comment line
        lineToRow = lambda line: map(float, string.split(line.replace(',','.'))) 
        res = map(lineToRow, file.readlines())  
        file.close()  # close file could be ommitted, reference
                      # counting should do during garbage collection, but...
        return res
#     except :
        print 'could not read file', file_name


#____________________________________________________________
#____________________________________________________________
class Error(Exception) :
    """generic exception of cma module"""
    pass


#____________________________________________________________
#____________________________________________________________
# 
def prettyprint(something):
    """ nicely formated print """
    import pprint 
    p = pprint.PrettyPrinter().pprint(something) # generates and instance PrettyPrinter

def pprint(something):
    """ alias for prettyprint """
    prettyprint(something)

def pp(something):
    """ alias for prettyprint """
    prettyprint(something)

#____________________________________________________________
#____________________________________________________________

# for testing purpose
if 1 < 3 and __name__ == "__main__":
    tic = time.time()
    res = fminCMA(felli, numpy.ones(10)*0.3, 1e-2, (1,), ftarget=1e-9, CMAdiagonal='10*N',
                  CMArankmualpha=[], tolfacupx=1e9, verb_log=10)
    # PlotCMAdata()
    # print res[2]['es'].sp.rankmualpha
    print 'elapsed time [s]:', round(time.time() - tic, 2)

