"""A simple simulation environment for running behavior trees on simulations."""

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
from typing import Any, List

from behaviors.common_behaviors import ParameterizedNode
from simulation.py_trees_interface import PyTree


@dataclass
class EnvParameters:
    """Data class for parameters for the environment."""

    seed: Any = None  # Random seed
    verbose: bool = False  # Extra prints
    py_tree_parameters: Any = None  # Parameters for executing the py tree
    sim_class: Any = None  # Simulation class
    sim_parameters: Any = None  # Parameters specific for the simulation
    fitness_func: Any = None  # Fitness function pointer
    fitness_coeff: Any = None  # Coefficients for the fitness function


class Environment:
    """Class defining the environment in which the individual operates."""

    def __init__(self, parameters: Any):
        self.par = parameters
        self.sim_class = self.par.sim_class
        self.fitness_function = self.par.fitness_func
        self.world_interface = None
        self.pytree = None

    def get_fitness(
        self,
        individual: List[ParameterizedNode],
        seed: int = None,
        make_video: bool = False,
        video_logdir: str = "filmtest"
    ) -> float:
        """Run the simulation and return the fitness."""
        if seed is not None:
            self.par.seed = seed

        self.world_interface = self.sim_class(
            self.par.sim_parameters, self.par.seed)
        if make_video:
            self.world_interface.set_make_video()
        pytree = PyTree(
            individual[:],
            parameters=self.par.py_tree_parameters,
            world_interface=self.world_interface
        )

        # run the Behavior Tree
        ticks, _ = pytree.run_bt()

        if make_video:
            self.world_interface.make_video(logdir=video_logdir)

        return self.fitness_function(
            self.world_interface, pytree, ticks, self.par.fitness_coeff, verbose=self.par.verbose)

    def step(self, individual: List[ParameterizedNode]):
        """Step the BT."""
        if self.world_interface is None:
            self.world_interface = self.sim_class(self.par.sim_parameters, self.par.seed)
        if self.pytree is None:
            self.pytree = PyTree(
                node_list=individual[:],
                parameters=self.par.py_tree_parameters,
                world_interface=self.world_interface
            )

        self.pytree.step_bt()

    def plot_individual(
        self,
        path: str,
        plot_name: str,
        individual: List[ParameterizedNode]
    ):
        """Save a graphical representation of the individual."""
        pytree = PyTree(individual[:], parameters=self.par.py_tree_parameters)
        pytree.save_fig(path, name=plot_name)
