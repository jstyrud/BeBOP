# pylint: disable=redefined-outer-name
"""
Main script for running maple benchmarks with behavior trees
constructed with planner and Bayesian Optimization
"""

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
import os.path
import pickle
from statistics import mean
import argparse
from copy import deepcopy
from dataclasses import dataclass
import random
import numpy as np
import datetime

import pandas
from gym import logger

from behaviors.behavior_tree import BT
from behaviors.common_behaviors import ParameterizedNode
from behaviors.behavior_lists import BehaviorLists
import simulation.environment as env
from simulation.maple_benchmarks import behavior_list_settings, robosuite_behaviors, fitness_function
import simulation.maple_benchmarks.robosuite_interface as sim
from simulation.maple_benchmarks.planner.world_interface import WorldInterface
from simulation.maple_benchmarks.planner import planner_behaviors
from simulation.py_trees_interface import PyTreeParameters

from bt_learning.planner import planner
from bt_learning.bo import bo
from bt_learning.gp import logplot
from bt_learning.cma_es import cma_es_interface


@dataclass
class HandlerSettings:
    """Data class for parameters for the BoHandler."""
    log_name: str = 'bo_test'             # Name of the log
    iterations: int = 100                 # Number of iterations to run
    runs_per_bt: int = 20                 # Maximum number of runs with different seeds per bt
    validation_runs: int = 20             # Number of validation to run
    hotstart: bool = False                # Hotstart from a previous run or start from scratch
    random_search: bool = False           # Use random step search instead of BO
    random_forest: bool = True            # Use random forest model instead of GP
    cma_es: bool = False                  # Use CMA-ES instead of BO
    cascaded: bool = False                # Run subtrees sequentially in cascaded learning


class BoHandler():
    """ Class ho handle Bayesian Optimization runs. Gets fitness and returns fitness and logs data """
    def __init__(self, bt, env_parameters, settings):
        self.bt = bt
        self.environment = env.Environment(env_parameters)
        self.type = env_parameters.sim_parameters.type
        self.behavior_lists = env_parameters.py_tree_parameters.behavior_lists
        self.log_name = settings.log_name
        self.log_folder = logplot.get_log_folder(settings.log_name)
        self.csv_file_name = self.log_folder + "/bo_log.csv"
        self.iterations = settings.iterations
        self.parameter_values = []
        self.fitness = []
        self.net_fitness = []
        self.steps = []
        self.validation_fitness = []
        self.n_validation_successes = []
        self.cumulative_steps = []
        self.best_fitness = -9999999999
        self.best_fitness_updated = False
        self.best_bt = deepcopy(bt)
        self.best_current_bt = deepcopy(bt)
        self.hotstart = settings.hotstart
        self.random_search = settings.random_search
        self.cma_es = settings.cma_es
        self.random_forest = settings.random_forest
        if settings.hotstart:
            self.parameter_values, self.fitness, self.net_fitness, \
                self.validation_fitness, self.n_validation_successes, \
                self.steps, self.cumulative_steps, self.best_fitness, self.best_bt = BoHandler.load_data(settings.log_name)
            self.bt = deepcopy(self.best_bt)
            self.best_current_bt = deepcopy(self.best_bt)
        else:
            logplot.clear_logs(settings.log_name)

        self.runs_per_bt = settings.runs_per_bt
        self.validation_runs = settings.validation_runs

        self.fix_set_categorical()
        self.current_bt = self.bt

    def fix_set_categorical(self):
        """
        Fixes the value of any categorical that already has a set value
        so that value doesn't change during learning
        """
        for node in self.bt:
            if isinstance(node, ParameterizedNode) and node.parameters is not None:
                for parameter in node.parameters:
                    if parameter.list_of_values != [] and parameter.value not in ['', 0.0]:
                        parameter.list_of_values = [parameter.value]

    def fix_conditions(self, behavior_type, threshold=None, value=None):
        """ Fixes the values of any conditions of the given type """
        for node in self.bt:
            if isinstance(node, ParameterizedNode) and node.behavior is behavior_type and \
                 node.parameters is not None:
                if threshold is not None:
                    node.parameters[-1].value = threshold
                for parameter in node.parameters:
                    if value is not None and parameter.value in ['', 0.0]:
                        parameter.value = value
                    parameter.min = parameter.value
                    parameter.max = parameter.value
                    if parameter.list_of_values != [] and parameter.value not in ['', 0.0]:
                        parameter.list_of_values = [parameter.value]

    def call_optimizer(self, bt, log_folder, iterations, add_priors=False):
        """ Call the optimization algorithm """
        if self.random_search:
            self.run_random_step_search(bt, iterations)
        elif self.cma_es:
            cma_es_interface.optimize_parameters(self.get_parameterized_nodes(bt), self.callback,
                                                 log_folder, iterations=iterations)
        else:
            bo.optimize_parameters(self.get_parameterized_nodes(bt), self.callback,
                                   log_folder, iterations=iterations, add_priors=add_priors,
                                   exp_name=self.type, random_forest=self.random_forest)

    def run_optimization(self, cascaded=False):
        """ Runs the actual optimization """
        if not cascaded:
            self.call_optimizer(self.bt, self.log_folder, self.iterations, add_priors=False)
        else:
            n = 1
            if self.hotstart:
                new_subtree = False
            else:
                new_subtree = True
            is_full_tree = False
            iterations = 0
            while not is_full_tree:
                self.best_fitness_updated = False
                bt = BT(self.bt, self.behavior_lists)
                if new_subtree:
                    self.best_fitness = -9999999999
                    iterations = 50

                self.current_bt, is_full_tree = bt.get_nth_subtree(n)

                if is_full_tree:
                    iterations = self.iterations
                    log_folder = self.log_folder
                else:
                    iterations += 50
                    log_folder = self.log_folder+"/subtree_"+str(n)
                    if new_subtree:
                        logplot.make_directory(log_folder)

                if n > 1:
                    self.call_optimizer(self.current_bt, log_folder, iterations, add_priors=True)
                else:
                    self.call_optimizer(self.current_bt, log_folder, iterations, add_priors=False)

                if not self.best_fitness_updated:
                    self.bt = deepcopy(self.best_bt)
                    if not is_full_tree:
                        # Use new values as priors in next subtree
                        bt = BT(self.bt, self.behavior_lists)
                        subtree, _ = bt.get_nth_subtree(n)
                        for node in subtree:
                            if isinstance(node, ParameterizedNode) and node.parameters is not None:
                                for parameter in node.parameters:
                                    # parameter.min = parameter.value
                                    # parameter.max = parameter.value
                                    parameter.use_prior = True
                    n += 1
                    new_subtree = True
                else:
                    new_subtree = False  # Not solved yet, redo subtree again

        self.plot()
        print("Best tree: \n" + str(self.best_bt))

    def run_parameters(self, parameters, seed=100, save_video=True, video_logdir="filmtest"):
        """ Run once with given set of parameters, save video and print fitness """
        parameter_index = 0
        for i in range(len(self.current_bt)):
            if isinstance(self.current_bt[i], ParameterizedNode) and self.current_bt[i].parameters is not None:
                for parameter in self.current_bt[i].parameters:
                    if parameter.min != parameter.max:
                        if isinstance(parameter.value, float):
                            parameter.value = round(parameters[parameter_index], 8)
                            parameter_index += 1
                        elif isinstance(parameter.value, tuple):
                            rounded_value = []
                            for _ in range(len(parameter.value)):
                                rounded_value.append(round(parameters[parameter_index], 8))
                                parameter_index += 1
                            parameter.value = tuple(rounded_value)

        fitness, net_fitness, steps, success = self.environment.get_fitness(self.current_bt, seed, save_video, video_logdir)

        print("Fitness:", fitness)
        print("Net fitness:", net_fitness)
        print("Steps:", steps)
        print("Success:", success)

    @staticmethod
    def get_parameterized_nodes(bt):
        """ Returns a list of only the parameterized nodes in the BT """
        parameterized_nodes = []
        for node in bt:
            if isinstance(node, ParameterizedNode):
                parameterized_nodes.append(node)
        return parameterized_nodes

    def callback(self):
        """
        Function for bo to callback to in order to get fitness
        Will also log the data
        """
        parameter_values = []
        for i in range(len(self.current_bt)):
            if isinstance(self.current_bt[i], ParameterizedNode) and self.current_bt[i].parameters is not None:
                for parameter in self.current_bt[i].parameters:
                    if isinstance(parameter.value, float):
                        parameter.value = round(parameter.value, 8)
                    elif isinstance(parameter.value, tuple):
                        rounded_value = []
                        for value in parameter.value:
                            rounded_value.append(round(value, 8))
                        parameter.value = tuple(rounded_value)
                parameters = self.current_bt[i].get_parameters()
                if parameters is not None:
                    parameter_values += parameters

        cumulative_steps = 0
        fitnesses = []
        net_fitnesses = []
        n_successes = 0
        for i in range(self.runs_per_bt):
            fitness, net_fitness, steps, success = self.environment.get_fitness(self.current_bt, 100+i)
            fitnesses.append(fitness)
            net_fitnesses.append(net_fitness)
            cumulative_steps += steps

            n_fitnesses = len(fitnesses)
            fitness_estimate = mean(fitnesses)
            margin = self.best_fitness - fitness_estimate
            std = np.std(fitnesses)
            if success:
                n_successes += 1

            df = pandas.DataFrame(data={"fitness": fitness,
                                        "mean fitness": fitness_estimate,
                                        "net_fitness": net_fitness,
                                        "margin": margin,
                                        "std": std,
                                        "steps": steps,
                                        "success": success,
                                        "n_successes": n_successes,
                                        "seed": str(i),
                                        "bt": str(self.current_bt)},
                                  index=[0])
            df.to_csv(self.csv_file_name,
                      mode='a',
                      sep=';',
                      index=False,
                      header=not os.path.exists(self.csv_file_name))

            if n_fitnesses >= 3 and self.best_fitness > fitness_estimate:
                #z = 1.28155 # Corresponds to 80% confidence
                #z = 1.43953 # Corresponds to 85% confidence
                #z = 1.64485 # Corresponds to 90% confidence
                z = 1.95996 # Corresponds to 95% confidence
                #z = 2.57583 # Corresponds to 99% confidence
                if z ** 2 * std ** 2 < margin ** 2 * n_fitnesses:
                    break

        fitness = mean(fitnesses)
        net_fitness = mean(net_fitnesses)

        self.parameter_values.append(parameter_values)
        self.fitness.append(fitness)
        self.net_fitness.append(net_fitness)
        self.steps.append(cumulative_steps)
        if len(self.cumulative_steps) > 0:
            self.cumulative_steps.append(self.cumulative_steps[-1] + cumulative_steps)
        else:
            self.cumulative_steps.append(cumulative_steps)

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_fitness_updated = True
            self.best_bt = deepcopy(self.bt)
            self.best_current_bt = deepcopy(self.current_bt)
            validation_fitness, n_successes = self.run_validation()
            self.validation_fitness.append(validation_fitness)
            self.n_validation_successes.append(n_successes)
        else:
            self.validation_fitness.append(self.validation_fitness[-1])
            self.n_validation_successes.append(self.n_validation_successes[-1])
        self.log_data()

        print(fitness, net_fitness, cumulative_steps)

        return fitness

    def run_validation(self):
        """ Runs the current tree on new seeds for validation and plotting """
        net_fitnesses = []
        n_successes = 0
        if self.validation_runs > 0:
            for i in range(self.validation_runs):
                _, net_fitness, _, success = self.environment.get_fitness(self.current_bt, 1337+i)

                net_fitnesses.append(net_fitness)
                if success:
                    n_successes += 1
        else:
            return 0.0, 0.0

        return mean(net_fitnesses), n_successes

    def run_random_step_search(self, bt, iterations):
        """
        Random step search:
        Adds gaussian noise to all parameters
        If new tree not is better than best tree, revert to best tree again
        Repeat
        """
        n_parameters = 0
        for node in bt:
            if isinstance(node, ParameterizedNode) and node.parameters is not None:
                for parameter in node.parameters:
                    if isinstance(parameter.step, float):
                        parameter.step = 0.0000001
                        n_parameters += 1
                    elif isinstance(parameter.step, tuple):
                        parameter.step = (0.0000001, 0.0000001, 0.0000001)
                        n_parameters += 3
        if iterations is None:
            iterations = 20 * n_parameters

        for _ in range(iterations):
            # Need to save random state and reload because seed is fixed in callback
            randomstate = random.getstate()
            np_randomstate = np.random.get_state()
            fitness = self.callback()
            random.setstate(randomstate)
            np.random.set_state(np_randomstate)
            if fitness < self.best_fitness:
                # Reset
                for i in range(len(bt)):
                    if isinstance(bt[i], ParameterizedNode) and bt[i].parameters is not None:
                        for j in range(len(bt[i].parameters)):
                            bt[i].parameters[j].value = self.best_current_bt[i].parameters[j].value

            for node in bt:
                if isinstance(node, ParameterizedNode):
                    node.randomize_parameters()

    def log_data(self):
        """ Save the log data for later retrieval """
        with logplot.open_file(self.log_folder + '/bo_data.pickle', 'wb') as f:
            pickle.dump((self.parameter_values, self.fitness, self.net_fitness,
                         self.validation_fitness, self.n_validation_successes,
                         self.steps, self.cumulative_steps, self.best_fitness, self.best_bt), f)

    @staticmethod
    def load_data(log_name):
        """ Load saved data for using for plotting, hotstart etc """
        with logplot.open_file(logplot.get_log_folder(log_name) + '/bo_data.pickle', 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def get_bo_data(logs):
        """ Retrieve logged bo data """
        fitness = []
        steps = []
        n_successes = []
        for log_name in logs:
            data = BoHandler.load_data(log_name)
            fitness.append(data[3])
            steps.append(data[6])
            n_successes.append(data[4])
            print(log_name + " validation successes: " + str(data[4][-1]) + ", n steps " + str(data[6][-1]))

        return fitness, n_successes, steps

    def plot(self):
        """ Plots the run"""
        plotpars = logplot.PlotParameters()
        plotpars.xlabel = 'Steps'
        plotpars.ylabel = 'Fitness'
        plotpars.x_step = 10
        plotpars.plot_horizontal = False
        plotpars.path = self.log_folder + '/fitness.pdf'

        fitness_logs, n_successes_logs, n_steps_logs = BoHandler.get_bo_data([self.log_name])

        logplot.plot_learning_curves([], plotpars, n_steps_logs, fitness_logs)

        # Success rate plot
        plotpars.y_scale = 5.0
        plotpars.ylabel = 'Success rate (%)'
        plotpars.path = self.log_folder + '/success_rate.pdf'
        logplot.plot_learning_curves([], plotpars, n_steps_logs, n_successes_logs)


def get_bo_handler(env_parameters, bo_settings, env_type="lift", fix_goal_condition=True):
    """ Set up and return BO handler """
    if env_type == "lift":
        world_interface = WorldInterface('lift', ['cube'])
        if fix_goal_condition:
            goals = [planner_behaviors.AtPos('at ', ['cube', 'none', '(0.0, 0.0, 0.1)', True], world_interface)]
        else:
            goals = [planner_behaviors.AtPos('at ', ['cube', 'none', 'unknown', True], world_interface)]
        env_parameters.py_tree_parameters.behavior_lists = \
            behavior_list_settings.get_behavior_list(['cube'],
                                                     at_pos_threshold=0.06,
                                                     random_step=True)
    elif env_type == "door":
        world_interface = WorldInterface('door', ['handle'])
        if fix_goal_condition:
            goals = [planner_behaviors.DoorOpen('door open', ['0.1'], world_interface)]
        else:
            goals = [planner_behaviors.DoorOpen('door open', [], world_interface)]
        env_parameters.py_tree_parameters.behavior_lists = behavior_list_settings.get_behavior_list(['handle'],
                                                                                                    has_door=True,
                                                                                                    angle_control=False,
                                                                                                    random_step=True)
    elif env_type == "pnp":
        world_interface = WorldInterface('pnp', ['can'])
        if fix_goal_condition:
            goals = [planner_behaviors.AtPos('at ', ['can', 'none', '(0.0, 0.15, 0.0)', False], world_interface)]
        else:
            goals = [planner_behaviors.AtPos('at ', ['can', 'none', 'unknown', False], world_interface)]
        env_parameters.py_tree_parameters.behavior_lists = \
            behavior_list_settings.get_behavior_list(['can'],
                                                     at_pos_threshold=0.07,
                                                     at_pos_free=True,
                                                     random_step=True)
    elif env_type == "wipe":
        world_interface = WorldInterface('wipe', ['centroid'])
        goals = [planner_behaviors.Wiped('wiped', ['centroid'], world_interface)]
        env_parameters.py_tree_parameters.behavior_lists = behavior_list_settings.get_behavior_list(['centroid'],
                                                                                                    random_step=True,
                                                                                                    angle_control=False,
                                                                                                    gripper_control=False,
                                                                                                    root_nodes=['s(', 'sm('],
                                                                                                    sequence_nodes=['s(', 'sm(']
                                                                                                    )
        env_parameters.py_tree_parameters.max_ticks = 300
    elif env_type == "stack":
        world_interface = WorldInterface('stack', ['red', 'green'])
        if fix_goal_condition:
            goals = [planner_behaviors.AtPos('at ', ['red', 'green', '(0.0, 0.0, 0.02)', False], world_interface)]
        else:
            goals = [planner_behaviors.AtPos('at ', ['red', 'green', 'unknown', False], world_interface)]
        env_parameters.py_tree_parameters.behavior_lists = behavior_list_settings.get_behavior_list(['red', 'green'],
                                                                                                    at_pos_free=True,
                                                                                                    random_step=True)
    elif env_type == "nut":
        world_interface = WorldInterface('nut', ['nut'])
        if fix_goal_condition:
            goals = [planner_behaviors.AtPos('at ', ['nut', 'none', '(0.125, -0.1, 0.0)', False], world_interface)]
        else:
            goals = [planner_behaviors.AtPos('at ', ['nut', 'none', 'unknown', False], world_interface)]
        env_parameters.py_tree_parameters.behavior_lists = behavior_list_settings.get_behavior_list(['nut'],
                                                                                                    at_pos_free=True,
                                                                                                    random_step=True)
    elif env_type == "cleanup":
        world_interface = WorldInterface('cleanup', ['spam', 'jello'], ['spam'])
        if fix_goal_condition:
            goals = [planner_behaviors.AtPos('at ', ['spam', 'none', '(-0.15, -0.15, 0.0)', False], world_interface),
                     planner_behaviors.AtPos('at ', ['jello', 'none', '(-0.15, 0.15, 0.0)', False], world_interface)]
        else:
            goals = [planner_behaviors.AtPos('at ', ['spam', 'none', 'unknown', False], world_interface),
                     planner_behaviors.AtPos('at ', ['jello', 'none', 'unknown', False], world_interface)]

        env_parameters.py_tree_parameters.behavior_lists = \
            behavior_list_settings.get_behavior_list(['spam', 'jello'],
                                                     at_pos_threshold=0.08,
                                                     at_pos_free=True,
                                                     random_step=True)
    elif env_type == "peg_ins":
        world_interface = WorldInterface('peg_ins', ['peg'])
        if fix_goal_condition:
            goals = [planner_behaviors.Inserted('at ', ['peg', 'none', '(0.0, -0.15, 0.1)'], world_interface)]
        else:
            goals = [planner_behaviors.Inserted('at ', ['peg'], world_interface)]
        env_parameters.py_tree_parameters.behavior_lists = behavior_list_settings.get_behavior_list(['peg'],
                                                                                                    random_step=True,
                                                                                                    large_object=True)

    string_bt, _ = planner.plan(world_interface, planner_behaviors, goals, BehaviorLists(sequence_nodes=['s(', 'sm(']))
    print(string_bt)
    env_parameters.py_tree_parameters.behavior_lists.convert_from_string(string_bt)
    env_parameters.sim_parameters.type = env_type

    if env_type == "door":
        # door task has a rather different workspace, update limits for reach.
        i = 0
        for node in string_bt:
            if isinstance(node, ParameterizedNode) and node.name == 'reach':
                if i == 0:
                    # First reach for handle turning can actually open the door by itself, limit range to avoid this "cheating"
                    node.parameters[1].min = (-0.1, -0.1, -0.1)
                    node.parameters[1].max = (0.1, 0.1, 0.1)
                    i = 1
                else:
                    # Second reach range is shifted since door is so far to the right
                    # and allow up and down in z direction
                    node.parameters[1].min = (node.parameters[1].min[0], node.parameters[1].min[1] + 0.15, -0.1)
                    node.parameters[1].max = (node.parameters[1].max[0], node.parameters[1].max[1] + 0.15, 0.1)
                    break

    bo_handler = BoHandler(string_bt, env_parameters, bo_settings)
    if fix_goal_condition:
        if env_type == "lift":
            bo_handler.fix_conditions(robosuite_behaviors.AtPos, threshold=0.06)
        elif env_type == "door":
            bo_handler.fix_conditions(robosuite_behaviors.DoorAngle)
        elif env_type == "pnp":
            bo_handler.fix_conditions(robosuite_behaviors.AtPosFree, threshold=0.07)
        elif env_type == "stack" or env_type == "nut":
            bo_handler.fix_conditions(robosuite_behaviors.AtPosFree)
        elif env_type == "cleanup":
            bo_handler.fix_conditions(robosuite_behaviors.AtPosFree, threshold=0.08)
        elif env_type == "peg_ins":
            bo_handler.fix_conditions(robosuite_behaviors.AtPos)

    return bo_handler


def setup_and_run_bo(env_parameters, bo_settings, env_type="lift", fix_goal_condition=True):
    """ Set up BO and run optimization """
    bo_handler = get_bo_handler(env_parameters, bo_settings, env_type, fix_goal_condition)
    bo_handler.run_optimization(cascaded=bo_settings.cascaded)


if __name__ == "__main__":
    logger.set_level(40)
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='door', help='The experiment to run')
    parser.add_argument('--repetitions', type=int, default=1, help='The number of times to repeat the experiment')
    parser.add_argument('--iterations', type=int, default=None, help='The number of bo iterations to run')
    parser.add_argument('--hotstart', action='store_true', help='Whether to hotstart the bo')
    parser.add_argument('--cascaded', action='store_true', help='Whether to run the cascaded bo')
    parser.add_argument('--repetition', type=int, default=0, help='The repetition to run')
    parser.add_argument('--prefix', type=str, default='', help='The prefix for the output folder')

    # Do a central parse_args
    args = parser.parse_args()
    if args.prefix == '':
        args.prefix = datetime.datetime.now().strftime("%y%m%d")

    py_tree_parameters = PyTreeParameters()
    py_tree_parameters.behaviors = robosuite_behaviors
    py_tree_parameters.max_ticks = 150
    sim_parameters = sim.RobosuiteParameters()
    env_parameters = env.EnvParameters()
    env_parameters.py_tree_parameters = py_tree_parameters
    env_parameters.sim_class = sim.RobosuiteInterface
    env_parameters.sim_parameters = sim_parameters
    env_parameters.fitness_func = fitness_function.compute_fitness
    env_parameters.fitness_coeff = fitness_function.Coefficients()
    env_parameters.verbose = False

    bo_settings = HandlerSettings(iterations=args.iterations,
                                  hotstart=args.hotstart,
                                  cascaded=args.cascaded)
    bo_settings.runs_per_bt = 20
    bo_settings.validation_runs = 20
    bo_settings.random_search = False
    bo_settings.cma_es = False

    if args.repetitions > 1:
        for i in range(args.repetitions):
            bo_settings.log_name = args.prefix + "/" + args.experiment + "/" + args.experiment + "_" + str(i)
            setup_and_run_bo(env_parameters, bo_settings, env_type=args.experiment)
    else:
        bo_settings.log_name = args.prefix + "/" + args.experiment + "/" + args.experiment + "_" + str(args.repetition)
        setup_and_run_bo(env_parameters, bo_settings, env_type=args.experiment)

    # bo_settings = HandlerSettings()
    # bo_settings.iterations = 1000
    # bo_settings.runs_per_bt = 1
    # bo_settings.validation_runs = 1
    # bo_settings.hotstart = False
    # bo_settings.cascaded = False
    # bo_settings.log_name = "test_run"
    # bo_handler = get_bo_handler(env_parameters, bo_settings, env_type="cleanup")
    # bo_handler.run_parameters([0.0, 0.0, 0.0, 0.0, -0.15, -0.15, 0.1, -0.15, 0.15, 0.0, 0.2])