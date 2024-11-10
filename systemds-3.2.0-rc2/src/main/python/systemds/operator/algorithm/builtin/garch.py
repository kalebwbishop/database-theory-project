# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

# Autogenerated By   : src/main/python/generator/generator.py
# Autogenerated From : scripts/builtin/garch.dml

from typing import Dict, Iterable

from systemds.operator import OperationNode, Matrix, Frame, List, MultiReturn, Scalar
from systemds.script_building.dag import OutputType
from systemds.utils.consts import VALID_INPUT_TYPES


def garch(X: Matrix,
          kmax: int,
          momentum: float,
          start_stepsize: float,
          end_stepsize: float,
          start_vicinity: float,
          end_vicinity: float,
          sim_seed: int,
          verbose: bool):
    """
     This is a builtin function that implements GARCH(1,1), a statistical model used in analyzing time-series data where the variance
     error is believed to be serially autocorrelated
    
     COMMENTS
     This has some drawbacks: slow convergence of optimization (sort of simulated annealing/gradient descent)
     TODO: use BFGS or BHHH if it is available (this are go to methods)
     TODO: (only then) extend to garch(p,q); otherwise the search space is way too big for the current method
    
    
    
    :param X: The input Matrix to apply Arima on.
    :param kmax: Number of iterations
    :param momentum: Momentum for momentum-gradient descent (set to 0 to deactivate)
    :param start_stepsize: Initial gradient-descent stepsize
    :param end_stepsize: gradient-descent stepsize at end (linear descent)
    :param start_vicinity: proportion of randomness of restart-location for gradient descent at beginning
    :param end_vicinity: same at end (linear decay)
    :param sim_seed: seed for simulation of process on fitted coefficients
    :param verbose: verbosity, comments during fitting
    :return: simulated garch(1,1) process on fitted coefficients
    :return: variances of simulated fitted process
    :return: onstant term of fitted process
    :return: 1-st arch-coefficient of fitted process
    :return: 1-st garch-coefficient of fitted process
    """

    params_dict = {'X': X, 'kmax': kmax, 'momentum': momentum, 'start_stepsize': start_stepsize, 'end_stepsize': end_stepsize, 'start_vicinity': start_vicinity, 'end_vicinity': end_vicinity, 'sim_seed': sim_seed, 'verbose': verbose}
    
    vX_0 = Matrix(X.sds_context, '')
    vX_1 = Matrix(X.sds_context, '')
    vX_2 = Scalar(X.sds_context, '')
    vX_3 = Scalar(X.sds_context, '')
    vX_4 = Scalar(X.sds_context, '')
    output_nodes = [vX_0, vX_1, vX_2, vX_3, vX_4, ]

    op = MultiReturn(X.sds_context, 'garch', output_nodes, named_input_nodes=params_dict)

    vX_0._unnamed_input_nodes = [op]
    vX_1._unnamed_input_nodes = [op]
    vX_2._unnamed_input_nodes = [op]
    vX_3._unnamed_input_nodes = [op]
    vX_4._unnamed_input_nodes = [op]

    return op
