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
from dataclasses import field
from typing import Any

from behaviors.behavior_tree import BT


@dataclass
class Coefficients:
    """Coefficients for tuning the fitness function."""

    position: float = -1.0
    pos_acc: float = 0.1
    depth: float = 0.0
    length: float = -0.01
    ticks: float = 0.0
    failed: float = -5.0
    timeout: float = -1.0
    hand_not_empty: float = 0.0
    targets: list = field(default_factory=list)


def compute_fitness(
    world_interface: Any,
    py_tree: BT,
    ticks: int,
    coeff: 'Coefficients' = None,
    verbose: bool = False
) -> float:
    """Retrieve values and compute fitness."""
    if coeff is None:
        coeff = Coefficients()

    depth = py_tree.bt.depth()
    length = py_tree.bt.length()

    fitness = coeff.length * length + coeff.depth * depth + coeff.ticks * ticks
    if verbose:
        print('Fitness from length:', fitness)

    if coeff.position != 0.0:
        for i in range(len(coeff.targets)):
            fitness += coeff.position * max(
                0, world_interface.distance(i, coeff.targets[i]) - coeff.pos_acc)
            if verbose:
                print('Fitness:', fitness)
                print(i, ': ', world_interface.get_object_position(i))

    if py_tree.failed:
        fitness += coeff.failed
        if verbose:
            print('Failed: ', fitness)
    if py_tree.timeout:
        fitness += coeff.timeout
        if verbose:
            print('Timed out: ', fitness)
    if coeff.hand_not_empty != 0.0 and world_interface.get_picked() is not None:
        fitness += coeff.hand_not_empty
        if verbose:
            print('Hand not empty: ', fitness)

    # Just to ensure that binary approximation doesn't affect fitness ranking
    return round(fitness, 10)
