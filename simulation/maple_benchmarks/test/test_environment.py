"""Test running complete trees for this task via the environment."""

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
import py_trees as pt
import simulation.environment as env
from simulation.maple_benchmarks import behavior_list_settings, robosuite_interface, fitness_function
from simulation.maple_benchmarks import robosuite_behaviors as behaviors
from simulation.py_trees_interface import PyTreeParameters


def test_incomplete_episode():
    """
    Runs an episode for less than maximum ticks because of BT success
    and makes sure this gives the same result as a complete run
    """
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    interface = robosuite_interface.RobosuiteInterface(robosuite_parameters, 0)

    grasp_behavior = behaviors.Grasp('grasp', ['cube', [0.0, 0.0, 0.0], 0.0], interface)

    reach_behavior = behaviors.Reach('reach', ['none', [0.0, 0.0, 0.1]], interface)

    bt = pt.composites.Sequence('Sequence', memory=False)
    bt.add_child(grasp_behavior)
    bt.add_child(reach_behavior)
    bt.tick_once()
    interface.step()
    bt.tick_once()
    interface.step()
    bt.tick_once()
    assert bt.status == pt.common.Status.SUCCESS

    fitness, affordance_penalty, path_length, success = interface.get_fitness()
    for _ in range(150 - path_length):
        bt.tick_once()
        interface.step()

    fitness_full_run, affordance_penalty_full_run, path_length, success_full_run = interface.get_fitness()

    assert path_length == 150
    assert fitness_full_run == fitness
    assert affordance_penalty_full_run == affordance_penalty
    assert fitness > 700
    assert success
    assert success_full_run


def test_lift():
    """ creates a whole manually constructed Behavior Tree to solve the lift scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['cube'])
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'lift'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['f(', 'cube at none (0.0, 0.0, 0.1) 0.06',
                        's(', 'grasp cube (0.0, 0.0, 0.0) 0.0', 'reach none (0.0, 0.0, 0.1)', ')', ')']
    behavior_lists.convert_from_string(individual)
    fitness_tuple = environment.get_fitness(individual, 0, False)
    assert fitness_tuple[0] > 450   # Penalized fitness
    assert fitness_tuple[1] > 700   # Raw fitness
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 5 * parameters.fitness_coeff.length)) < 0.001


def test_door():
    """ creates a whole manually constructed Behavior Tree to solve the lift scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['handle'], angle_control=False, has_door=True)
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'door'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['f(', 'door angle > 0.1?',
                        's(', 'f(', 'handle angle > 1.0?',
                                    's(', 'grasp handle (0.0, 0.02, 0.0) 0.0!',
                                          'reach handle (0.1, 0.0, -0.1)!', ')', ')',
                                    'reach handle (-0.15, 0.25, -0.1)!', ')', ')']
    
    behavior_lists.convert_from_string(individual)
    for i in range(5):
        fitness_tuple = environment.get_fitness(individual, i, False)
        print(fitness_tuple)
    for i in range(5):
        fitness_tuple = environment.get_fitness(individual, 100+i, False)
        print(fitness_tuple)

    assert fitness_tuple[0] > 150
    assert fitness_tuple[1] > 600
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 9 * parameters.fitness_coeff.length)) < 0.001


def test_pnp():
    """ creates a whole manually constructed Behavior Tree to solve the pnp scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['can'], at_pos_free=True)
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'pnp'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['f(', 'can at none (0.0, 0.15, 0.0) 0.07',
                        's(', 'grasp can (0.0, 0.0, 0.01) 0.0', 'reach none (0.0, 0.15, 0.15)', 'open', ')', ')']

    behavior_lists.convert_from_string(individual)
    fitness_tuple = environment.get_fitness(individual, 0, False)
    print(fitness_tuple)
    assert fitness_tuple[0] > 300
    assert fitness_tuple[1] > 600
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 6 * parameters.fitness_coeff.length)) < 0.001


def test_wipe():
    """ creates a whole manually constructed Behavior Tree to solve the lift scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['centroid'], angle_control=False)
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'wipe'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['sm(', 'push centroid (-0.15, -0.15, 0.0) 0.2',
                         'push centroid (-0.15, 0.15, 0.0) 0.2',
                         'push centroid (0.15, 0.15, 0.0) 0.2',
                         'push centroid (0.15, -0.15, 0.0) 0.2',
                  ')']
    behavior_lists.convert_from_string(individual)
    fitness_tuple = environment.get_fitness(individual, 0, False)

    assert fitness_tuple[0] > 650   # Penalized fitness
    assert fitness_tuple[1] > 900   # Raw fitness
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 5 * parameters.fitness_coeff.length)) < 0.001


def test_stack():
    """ creates a whole manually constructed Behavior Tree to solve the stack scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['red', 'green'], at_pos_free=True)
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'stack'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['f(', 'red at green (0.0, 0.0, 0.02)',
                        's(', 'grasp red (0.0, 0.0, 0.0) 0.0', 'reach green (0.0, 0.0, 0.05)', 'open', ')', ')']

    behavior_lists.convert_from_string(individual)
    fitness_tuple = environment.get_fitness(individual, 0, False)

    assert fitness_tuple[0] > 650
    assert fitness_tuple[1] > 900
    assert fitness_tuple[3]
    # Some small affordance penalty could not be avoided in this case without lowering the step
    assert abs(fitness_tuple[1] - fitness_tuple[0] + 6 * parameters.fitness_coeff.length) < 4


def test_nut():
    """ creates a whole manually constructed Behavior Tree to solve the nut scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['nut'], at_pos_free=True)
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'nut'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.py_tree_parameters.successes_required = 2
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['f(', 'nut at none (0.125, -0.1, 0.0)',
                        's(', 'grasp nut (0.0, 0.05, 0.0) 0.0', 'reach none (0.1, -0.1, 0.15)', 'open', ')', ')']
    behavior_lists.convert_from_string(individual)
    fitness_tuple = environment.get_fitness(individual, 0, False)
    print(fitness_tuple)
    assert fitness_tuple[0] > 200
    assert fitness_tuple[1] > 500
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 6 * parameters.fitness_coeff.length)) < 0.001

    fitness_tuple = environment.get_fitness(individual, 1, False)
    assert fitness_tuple[0] > 200
    assert fitness_tuple[1] > 500
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 6 * parameters.fitness_coeff.length)) < 25


def test_cleanup():
    """ creates a whole manually constructed Behavior Tree to solve the cleanup scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['spam', 'jello'], at_pos_free=True)
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'cleanup'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.py_tree_parameters.successes_required = 2
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['s(', 'f(', 'spam at none (-0.15, -0.15, 0.0) 0.08',
                              's(', 'grasp spam (0.0, 0.0, 0.0) 0.0', 'reach none (-0.15, -0.15, 0.1)', 'open', ')', ')',
                        'f(', 'jello at none (-0.15, 0.15, 0.0) 0.08',
                              'push jello (-0.15, 0.15, 0.0) 0.2', ')',
                  ')']

    behavior_lists.convert_from_string(individual)
    fitness_tuple = environment.get_fitness(individual, 0, False)
    assert fitness_tuple[0] > 400
    assert fitness_tuple[1] > 800
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 10 * parameters.fitness_coeff.length)) < 0.001


def test_peg_ins():
    """ creates a whole manually constructed Behavior Tree to solve the peg_ins scenario """
    behavior_lists = behavior_list_settings.get_behavior_list(['peg'])
    robosuite_parameters = robosuite_interface.RobosuiteParameters()
    robosuite_parameters.type = 'peg_ins'
    parameters = env.EnvParameters()
    parameters.py_tree_parameters = PyTreeParameters()
    parameters.py_tree_parameters.behaviors = behaviors
    parameters.py_tree_parameters.behavior_lists = behavior_lists
    parameters.py_tree_parameters.successes_required = 2
    parameters.sim_class = robosuite_interface.RobosuiteInterface
    parameters.sim_parameters = robosuite_parameters
    parameters.fitness_func = fitness_function.compute_fitness
    parameters.fitness_coeff = fitness_function.Coefficients()
    environment = env.Environment(parameters)

    individual = ['f(', 'peg at none (0.0, -0.15, 0.1)',
                        's(', 'f(', 'aligned peg',
                                    's(', 'grasp peg (-0.1, 0.0, 0.0) 0.0',
                                          'atomic peg (0.0, 0.05, 0.1) -1.57 0.0', ')', ')',
                              'atomic peg (0.0, -0.15, 0.1) -1.57 0.0', ')', ')']

    behavior_lists.convert_from_string(individual)
    fitness_tuple = environment.get_fitness(individual, 0, False)
    assert fitness_tuple[0] > 150
    assert fitness_tuple[1] > 600
    assert fitness_tuple[3]
    assert round(abs(fitness_tuple[1] - fitness_tuple[0] + 9 * parameters.fitness_coeff.length)) < 0.001


if __name__ == "__main__":
    test_door()
