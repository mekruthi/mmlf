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

"""Module that implements the partially observable double pole balancing."""
# Refactored: 2010-01-21 by Jan Hendrik Metzen

from mmlf.framework.spaces import StateSpace
from mmlf.framework.protocol import EnvironmentInfo
from mmlf.worlds.double_pole_balancing.environments.double_pole_balancing_environment \
       import DoublePoleBalancingEnvironment\


class PODoublePoleBalancingEnvironment(DoublePoleBalancingEnvironment):
    """ The partially observable double pole balancing environment
    
    In the partially observable double pole balancing environment, 
    the task of the agent is to control a cart such that two poles which are mounted
    on the cart stay in a nearly vertical position (to balance them). At the same 
    time, the cart has to stay in a confined region. In contrast to the fully
    observable double pole balancing environment, the agent only observes the
    current position of cart and the two poles but not their velocities.
    This renders the problem to be not markovian.
    
    The agent can apply in every time step a force between -10N and 10N in order to
    accelerate the car. Thus the action space is one-dimensional and continuous. 
    The state consists of the cart's current position and velocity as well as the
    poles' angles and angular velocities. Thus, the state space is six-dimensional
    and continuous.
    
    The config dict of the environment expects the following parameters: 
    
    **CONFIG DICT** 
        :GRAVITY: : The gravity force. Benchmark default "-9.8".    
        :MASSCART: : The mass of the cart. Benchmark default "1.0".
        :TAU: : The time step between two commands of the agent. 
                Benchmark default "0.02"                         
        :MASSPOLE_1: : The mass of pole 1. Benchmark default "0.1"
        :MASSPOLE_2: : The mass of pole 2. Benchmark default "0.01"
        :LENGTH_1: : The length of pole 1. Benchmark default "0.5"
        :LENGTH_2: : The length of pole 2. Benchmark default "0.05"
        :MUP: : Coefficient of friction of the poles' hinges.
                Benchmark default "0.000002"
        :MUC: : Coefficient that controls friction. Benchmark default "0.0005"
        :INITIALPOLEANGULARPOSITION1: : Initial angle of pole 1. 
                                        Benchmark default "4.0"
        :MAXCARTPOSITION: : The maximal distance the cart is allowed to move away
                            from its start position. Benchmark default "2.4"
        :MAXPOLEANGULARPOSITION1: : Maximal angle pole 1 is allowed to take on. 
                                    Benchmark default "36.0"
        :MAXPOLEANGULARPOSITION2: : Maximal angle pole 2 is allowed to take on. 
                                    Benchmark default "36.0"
        :MAXSTEPS: : The number of steps the agent must balance the poles. 
                     Benchmark default "100000"
    """

    def __init__(self, useGUI, *args, **kwargs):
        
        self.environmentInfo = \
            EnvironmentInfo(versionNumber="0.3",
                            environmentName="Partially Observable Double Pole Balancing",
                            discreteActionSpace=False, episodic=True,
                            continuousStateSpace=True,
                            continuousActionSpace=True, stochastic=False)

        super(PODoublePoleBalancingEnvironment, self).__init__(useGUI=useGUI, *args, **kwargs)
        
        #The state space of partially observable double pole balancing
        oldStyleStateSpace =   {"cartPosition": ("continuous", [(-1.0, 1.0)]),
                                "poleAngularPosition1": ("continuous", [(-1.0, 1.0)]),
                                "poleAngularPosition2": ("continuous", [(-1.0, 1.0)])
                                }

        self.stateSpace = StateSpace()
        self.stateSpace.addOldStyleSpace(oldStyleStateSpace, limitType="soft")
        
        # The name of the state dimensions that are send to the agent.
        # NOTE: The ordering of the state dimensions is important!
        self.stateNameList = ["cartPosition", "poleAngularPosition1",
                              "poleAngularPosition2"]

EnvironmentClass = PODoublePoleBalancingEnvironment
EnvironmentName = "Partial Observable Double Pole Balancing"
