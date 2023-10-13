"""Task dependent fitness function."""

# Copyright (c) 2023, ABB
# All rights reserved.
#
# Redistribution and use in source and binary forms, with
# or without modification, are permitted provided that
# the following conditions are met:
#
#   * Redistributions of source code must retain the
#     above copyright notice, this list of conditions
#     and the following disclaimer.
#   * Redistributions in binary form must reproduce the
#     above copyright notice, this list of conditions
#     and the following disclaimer in the documentation
#     and/or other materials provided with the
#     distribution.
#   * Neither the name of ABB nor the names of its
#     contributors may be used to endorse or promote
#     products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from dataclasses import dataclass
from typing import Any

from behaviors.behavior_tree import BT
from simulation import fitness_function


@dataclass
class Coefficients(fitness_function.Coefficients):
    """Define coefficients for tuning the fitness function."""
    position: float = 0.0
    length: float = 0.0
    failed: float = -500.0
    timeout: float = 0.0


def compute_fitness(
    world_interface: Any,
    py_tree: BT,
    ticks: int,
    coeff: Coefficients = None,
    verbose: bool = False
) -> float:
    """Retrieve values and compute fitness."""
    if coeff is None:
        coeff = Coefficients()

    bt_fitness = fitness_function.compute_fitness(world_interface, py_tree, ticks, coeff, verbose)

    dense_fitness, affordance_penalty, n_steps, success = world_interface.get_fitness()

    fitness = bt_fitness + dense_fitness - affordance_penalty
    if verbose:
        print('bt_fitness:', bt_fitness)
        print('dense_fitness:', dense_fitness)
        print('affordance_penalty:', affordance_penalty)
        print('fitness:', fitness)
        print('success:', success)

    # Rounding just to ensure that numerical problems don't affect fitness ranking
    return (round(fitness, 10), round(dense_fitness, 10), n_steps, success)
