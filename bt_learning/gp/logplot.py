"""Handle logs and plots for learning."""

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
from copy import copy
from dataclasses import dataclass
import datetime
import json
import os
import pickle
import platform
import random
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np

if platform.python_implementation() == 'PyPy':
    import pypy_matplotlib as matplotlib  # pragma: no cover # pylint: disable=import-error
else:
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy import interpolate

import pandas
from pandas.plotting import parallel_coordinates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

matplotlib.rcParams['pdf.fonttype'] = 42


def moving_average(data, n=3):
    """ Applies a moving average filter of n steps onto the input data"""
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[n:] *= 1.0 / n
    for i in range(1, n):
        ret[i] *= 1.0 / (i + 1)
    return ret

def almost_equal(x, y, epsilon=0.001):
    """ Returns true if x and y are almost equal"""
    return abs(x - y) < epsilon

def open_file(path: str, mode: Any) -> Any:
    """
    Attempt to open file at path.

    Tried up to max_attempts times because of intermittent permission errors on windows.
    """
    max_attempts = 100
    f = None
    for _ in range(max_attempts):  # pragma: no branch
        try:
            f = open(path, mode)  # pylint: disable=unspecified-encoding
        except PermissionError:  # pragma: no cover
            continue
        break
    return f


def make_directory(path: str) -> None:
    """
    Attempt to create directory at path.

    Tried up to max_attempts times because of intermittent permission errors on windows.
    """
    max_attempts = 100
    for _ in range(max_attempts):  # pragma: no branch
        try:
            os.mkdir(path)
        except PermissionError:  # pragma: no cover
            continue
        break


def get_log_folder(log_name: str, add_time: bool = False) -> str:
    """Return log folder as string."""
    if not os.path.exists('logs'):
        os.mkdir('logs')  # pragma: no cover
    folder_name = 'logs/'
    if add_time:
        folder_name += datetime.datetime.now().strftime('%y%m%d-%H%M%S') + '_'
    folder_name += log_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def clear_logs(log_name: str) -> None:
    """Clear previous log folders of same same."""
    log_folder = get_log_folder(log_name)
    try:
        shutil.rmtree(log_folder)
    except FileNotFoundError:  # pragma: no cover
        pass

    make_directory(log_folder)
    fitness_log_path = log_folder + '/fitness_log.txt'
    population_log_path = log_folder + '/population_log.txt'
    open(fitness_log_path, 'x', encoding='utf-8')
    open(population_log_path, 'x', encoding='utf-8')


def clear_after_generation(log_name: str, generation: int) -> None:
    """Clear fitness and population logs after given generation."""
    with open_file(get_log_folder(log_name) + '/fitness_log.txt', 'r') as f:
        lines = f.readlines()
    with open_file(get_log_folder(log_name) + '/fitness_log.txt', 'w') as f:
        for i in range(generation + 1):
            f.write(lines[i])
    with open_file(get_log_folder(log_name) + '/population_log.txt', 'r') as f:
        lines = f.readlines()
    with open_file(get_log_folder(log_name) + '/population_log.txt', 'w') as f:
        for i in range(generation + 1):
            f.write(lines[i])


def log_best_individual(log_name: str, best_individual: Any):
    """Save the best individual."""
    with open_file(get_log_folder(log_name) + '/best_individual.pickle', 'wb') as f:
        pickle.dump(best_individual, f)


def log_fitness(log_name: str, fitness: float) -> None:
    """Log fitness of all individuals."""
    with open_file(get_log_folder(log_name) + '/fitness_log.txt', 'a') as f:
        f.write(f'{fitness}\n')


def log_diversity(
    log_name: str,
    diversity: float,
    n_gen: int,
    exchanged: bool
) -> None:
    """
    Log diversity of all individuals.

    Records the current generation count with n_gen.
    exchanged == True when the diversity is measured after exchanging
    """
    with open_file(get_log_folder(log_name) + '/diversity_log.txt', 'a') as f:
        f.write(f'generation: {n_gen}, exchanged: {exchanged}, {diversity}\n')


def log_best_fitness_episodes(log_name: str, fitness: float, n_episodes: int) -> None:
    """Log fitness of the best individual and the corresponding episodes."""
    with open_file(get_log_folder(log_name) + '/fitness_log_episodes.txt', 'a') as f:
        f.write(f'{n_episodes}\n{fitness}\n')


def log_tracking(
    log_name: str,
    num_exchange: int,
    num_from_exchange: int,
    num_from_replication: int,
    num_from_crossover: int,
    num_from_mutation: int
):
    """Log the counters that keep tracks of the methods that generate the new best individual."""
    with open_file(get_log_folder(log_name) + '/tracking_log.txt', 'a') as f:
        f.write(f'Exchange count: {num_from_exchange} / {num_exchange}\n')
        f.write(
            f'GP count: replication: {num_from_replication} crossover: {num_from_crossover}'
            f' mutation: {num_from_mutation}\n'
        )


def log_fitness_exchange(
    log_name: str,
    fitness: float,
    n_gen: int,
    exchanged: bool
) -> None:
    """
    Log fitness of all individuals.

    The function records the current generation count with n_gen.
    Set exchanged to True when the diversity is measured after exchanging.
    """
    with open_file(get_log_folder(log_name) + '/fitness_exchange_log.txt', 'a') as f:
        f.write(f'generation: {n_gen}, exchanged: {exchanged}, {fitness}\n')


def log_best_fitness(log_name: str, best_fitness: float) -> None:
    """Log best fitness of each generation."""
    with open_file(get_log_folder(log_name) + '/best_fitness_log.pickle', 'wb') as f:
        pickle.dump(best_fitness, f)


def log_best_net_fitness(log_name: str, best_net_fitness: float) -> None:
    """
    Log best fitness of each generation. This means only counting task fitness and
    excluding things like policy complexity or action affordances
    """
    with open_file(get_log_folder(log_name) + '/best_net_fitness_log.pickle', 'wb') as f:
        pickle.dump(best_net_fitness, f)


def log_n_episodes(log_name: str, n_episodes: int) -> None:
    """Log number of episodes."""
    with open_file(get_log_folder(log_name) + '/n_episodes_log.pickle', 'wb') as f:
        pickle.dump(n_episodes, f)


def log_n_steps(log_name: str, n_steps: int) -> None:
    """Log number of steps."""
    with open_file(get_log_folder(log_name) + '/n_steps_log.pickle', 'wb') as f:
        pickle.dump(n_steps, f)


def log_population(log_name: str, population: List[Any]) -> None:
    """Log full population of the generation."""
    with open_file(get_log_folder(log_name) + '/population_log.txt', 'a') as f:
        f.write(f'{population}\n')


def log_last_population(log_name: str, population: List[Any]) -> None:
    """Log current population as pickle object."""
    with open_file(get_log_folder(log_name) + '/population.pickle', 'wb') as f:
        pickle.dump(population, f)


def log_settings(log_name: str, settings: Dict, baseline: Any) -> None:
    """Log settings used for the run."""
    with open_file(get_log_folder(log_name) + '/settings.txt', 'w') as f:
        for key, value in vars(settings).items():
            f.write(key + ' ' + str(value) + '\n')
        f.write('Baseline: ' + str(baseline) + '\n')


def log_state(
    log_name: str,
    randomstate: Any,
    np_randomstate: float,
    generation: int
) -> None:
    """Log the current random state and generation number."""
    with open_file(get_log_folder(log_name) + '/states.pickle', 'wb') as f:
        pickle.dump(randomstate, f)
        pickle.dump(np_randomstate, f)
        pickle.dump(generation, f)


def get_best_fitness(log_name: str) -> float:
    """Get the best fitness list from the given log."""
    with open_file(get_log_folder(log_name) + '/best_fitness_log.pickle', 'rb') as f:
        best_fitness = pickle.load(f)
    return best_fitness


def get_best_net_fitness(log_name: str) -> float:
    """Get the best net fitness list from the given log."""
    with open_file(get_log_folder(log_name) + '/best_net_fitness_log.pickle', 'rb') as f:
        best_net_fitness = pickle.load(f)
    return best_net_fitness


def get_n_episodes(log_name: str) -> int:
    """Get the list of n_episodes from the given log."""
    with open_file(get_log_folder(log_name) + '/n_episodes_log.pickle', 'rb') as f:
        n_episodes = pickle.load(f)
    return n_episodes


def get_n_steps(log_name: str) -> int:
    """Get the list of n_steps from the given log."""
    with open_file(get_log_folder(log_name) + '/n_steps_log.pickle', 'rb') as f:
        n_steps = pickle.load(f)
    return n_steps


def get_state(log_name: str) -> Tuple[Any, float, int]:
    """Get the random state and generation number."""
    with open_file(get_log_folder(log_name) + '/states.pickle', 'rb') as f:
        randomstate = pickle.load(f)
        np_randomstate = pickle.load(f)
        generation = pickle.load(f)
    return randomstate, np_randomstate, generation


def get_last_population(log_name: str) -> List[Any]:
    """Get the last population list from the given log."""
    with open_file(get_log_folder(log_name) + '/population.pickle', 'rb') as f:
        population = pickle.load(f)
    return population


def get_best_individual(log_name: str) -> Any:
    """Return the best individual from the given log."""
    with open_file(get_log_folder(log_name) + '/best_individual.pickle', 'rb') as f:
        best_individual = pickle.load(f)
    return best_individual


def plot_fitness(log_name, fitness, x_values=None, best_net_fitness=None, x_label="Episodes") -> None:
    """Plot fitness over iterations or individuals."""
    if x_values is not None:
        plt.plot(x_values, fitness, label="Fitness")
        plt.xlabel(x_label)
    else:
        plt.plot(fitness, label="Fitness")
        plt.xlabel('Generation')
    if best_net_fitness is not None:
        if x_values is not None:
            plt.plot(x_values, best_net_fitness, label="Net fitness")
            plt.xlabel(x_label)
        else:
            plt.plot(best_net_fitness, label="Net fitness")
            plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(get_log_folder(log_name) + '/Fitness_' + x_label + '.png')
    plt.close()


# pylint: disable=too-many-instance-attributes
@dataclass
class PlotParameters:
    """Data class for parameters for plotting."""

    plot_mean: bool = True                # Plot the mean of the logs
    mean_color: str = None                 # Color for mean curve
    plot_std: bool = True                 # Plot the standard deviation
    std_color: str = None                  # Color of the std fill
    plot_minmax: bool = False             # Plots minmax instead of std, should not be combined
    plot_ind: bool = False                # Plot each individual log
    ind_color: str = 'aquamarine'         # Ind color
    label: str = ''                       # Label name
    title: str = ''                       # Plot title
    xlabel: str = ''                      # Label of x axis
    fig_width: int = 0                    # Width of figure
    fig_height: int = 0                   # Height of figure
    endx: int = 0                         # Stop plotting after this point
    x_max: int = 0                        # Upper limit of x axis
    x_step: int = 1                       # Plot resolution in x
    y_max: int = -1000                    # Upper limit of y axis
    y_min: int = 1000                     # Lower limit of y axis
    y_scale: float = 1.0                  # Scale y values with this
    y_moving_average: int = 0             # Number of steps in moving average filter (0 = no filter)
    extend_x: int = 0                     # Extend all logs until this minimum x
    ylabel: str = ''                      # Label of y axis
    extrapolate_y: bool = False           # Extrapolate y as constant to x_max
    logarithmic_x: bool = False           # Logarithmic x scale
    linear_x_threshold: float = 100000    # x is still linear up to this value
    logarithmic_y: bool = False           # Logarithmic y scale
    plot_horizontal: bool = True          # Plot thin horizontal line
    horizontal: float = 0                 # Horizontal value to plot
    horizontal_label: str = ''            # Label of horizontal line
    horizontal_linestyle: str = 'dashed'  # Style of horizontal line
    horizontal_color: str = 'b'           # Color of horizontal line
    linewidth: float = 3.0                # Line width
    legend_position: str = 'lower right'  # Position of legend
    save_fig: bool = True                 # Save figure. If false, more plots is possible.
    path: str = 'logs/plot.pdf'           # Path to save log
    net_fitness: bool = True              # Use net fitness, i.e. counting task fitness
    x_axis_n_steps: bool = False          # Use number of steps as x axis


def plot_learning_curves(logs: List[str], parameters: PlotParameters, x_logs=None, y_logs=None) -> None:
    # pylint: disable=too-many-branches, too-many-statements, too-many-locals
    """Plot mean and standard deviation of a number of logs in the same figure."""
    if (parameters.fig_width > 0 and parameters.fig_height > 0):
        plt.rcParams['figure.figsize'] = (parameters.fig_width, parameters.fig_height)
    fitness = []
    x_values = []
    if logs:
        for log_name in logs:
            if parameters.net_fitness:
                fitness.append(get_best_net_fitness(log_name))
            else:
                fitness.append(get_best_fitness(log_name))
            if parameters.x_axis_n_steps:
                x_values.append(get_n_steps(log_name))
            else:
                x_values.append(get_n_episodes(log_name))
        n_logs = len(logs)
    else:
        for x_log in x_logs:
            x_values.append(copy(x_log))
        for y_log in y_logs:
            fitness.append(copy(y_log))
        n_logs = len(x_values)

    if parameters.extend_x > 0:
        # Extend until this minimum x, assuming shorter logs are stopped because
        # they have converged there is no difference to end result
        for i in range(n_logs):
            if x_values[i][-1] < parameters.extend_x:
                fitness[i].append(fitness[i][-1])
                x_values[i].append(parameters.extend_x)

    startx = max(x_log[0] for x_log in x_values)
    if parameters.endx == 0:
        endx = min(x_log[-1] for x_log in x_values)
    else:
        endx = parameters.endx

    for i in range(n_logs):
        fitness[i] = np.array(fitness[i]) * parameters.y_scale
        x_values[i] = np.array(x_values[i])

    if parameters.extrapolate_y:
        x = np.arange(startx, parameters.x_max + 1, parameters.x_step)
    else:
        x = np.arange(startx, endx + 1, parameters.x_step)

    if parameters.plot_horizontal:
        plt.plot([0, parameters.x_max],
                 [parameters.horizontal, parameters.horizontal],
                 color=parameters.horizontal_color, linestyle=parameters.horizontal_linestyle,
                 linewidth=parameters.linewidth, label=parameters.horizontal_label)

    y = np.zeros((len(x), n_logs))
    for i in range(n_logs):
        f = interpolate.interp1d(x_values[i], fitness[i], bounds_error=False)
        y[:, i] = f(x)
        if parameters.extrapolate_y:
            n_extrapolated = int(parameters.x_max - x_values[i][-1])
            if n_extrapolated > 0:
                left = y[:x_values[i][-1] - x_values[i][0] + 1, i]
                y[:, i] = np.concatenate((left, np.full(n_extrapolated, left[-1])))
        if parameters.plot_ind:
            plt.plot(x, y[:, i], color=parameters.ind_color, linestyle='dashed', linewidth=parameters.linewidth)

    y_mean = np.mean(y, axis=1)

    if parameters.y_moving_average > 0:
        y_mean = moving_average(y_mean, parameters.y_moving_average)

    if parameters.plot_mean:
        plt.plot(x, y_mean, color=parameters.mean_color, label=parameters.label, linewidth=parameters.linewidth)

    if parameters.plot_std:
        y_std = np.std(y, axis=1)
        if parameters.y_moving_average > 0:
            y_std = moving_average(y_std, parameters.y_moving_average)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=.2, color=parameters.std_color)
    if parameters.plot_minmax:
        maxcurve = np.max(y, axis=1)
        mincurve = np.min(y, axis=1)
        plt.fill_between(x, mincurve, maxcurve, alpha=.1, color=parameters.std_color)

    if parameters.legend_position is not None:
        plt.legend(loc=parameters.legend_position, facecolor='white')
    plt.xlabel(parameters.xlabel)
    if parameters.x_max > 0:
        plt.xlim(0, parameters.x_max)
    if parameters.y_max > -1000:
        plt.ylim(top=parameters.y_max)
    if parameters.y_min < 1000:
        plt.ylim(bottom=parameters.y_min)

    if parameters.logarithmic_y:
        plt.yscale('symlog')
        plt.yticks([0, -1, -10, -100], ('0', '-1', '-10', '-100'))
    if parameters.logarithmic_x:
        plt.xscale('symlog', linthresh=parameters.linear_x_threshold)

    plt.ylabel(parameters.ylabel)
    plt.title(parameters.title)
    plt.grid(True)
    if parameters.save_fig:
        plt.tight_layout(pad=0.05)
        plt.savefig(parameters.path, format='pdf', dpi=300)
        plt.close()


def plot_parallel_coordinates(directory: str, parameters: PlotParameters, fitness_percentile = True) -> None:
    """Plot parallel coordinates of an experiment."""
    plt.rcParams['figure.figsize'] = (parameters.fig_width, parameters.fig_height)
    if directory[-1] != '/':
        directory += '/'

    # Read json file
    with open(directory + "scenario.json", 'r') as json_file:
        json_file = json.load(json_file)
    exp = json_file["application_name"]

    # Fetch the parameter ranges
    ranges = {}
    for param_name in json_file["input_parameters"]:
        param = json_file["input_parameters"][param_name]
        if len(param["values"]) != 2:
            print("Error: parameter {} has no range for rescaling".format(param_name))
            continue
        ranges[param_name] = param["values"]

    # Read hypermapper csv file
    data = pandas.read_csv(directory + "{}_output_samples.csv".format(exp), sep=',')

    # Rescale parameters from -1 to 1
    for param_name in ranges:
        data[param_name] = data[param_name].apply(lambda x: (x - ranges[param_name][0]) / (ranges[param_name][1] - ranges[param_name][0]) * 2 - 1)
    # Fetch lowest and highest fitness
    fitness_range = (data["fitness"].min(), data["fitness"].max())

    num_classes = 100
    # divide the samples based their fitness percentile
    if fitness_percentile:
        classes = np.linspace(fitness_range[0], fitness_range[1], num_classes)
        data['class'] = np.digitize(data['fitness'], classes) - 1 # Assign class to each row in the dataframe based on the fitness column
    else:
        # divide classes based on the sample percentile
        data['class'] = pandas.qcut(data['fitness'], num_classes, labels=False, duplicates='drop')

    # order dataframe from highest to lowest fitness. Colormap needs to fit
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=num_classes)
    data = data.sort_values(by=['class'], ascending=False)

    # print the columns class and fitness of the first 20 and last 20 rows 
    print(data[['class', 'fitness']].head(2))
    print(data[['class', 'fitness']].tail(2))

    for param_name in ranges:
        data[param_name] = data[param_name].apply(lambda x: x + random.uniform(-0.05, 0.00) if almost_equal(x, -1.0) else x)
        data[param_name] = data[param_name].apply(lambda x: x + random.uniform(-0.00, 0.05) if almost_equal(x, 1.0) else x)

    # Remove columns that are not parameters
    data = data.drop(columns=['timestamp', 'fitness'])

    parallel_coordinates(data, 'class', colormap=cmap)

    plt.xticks(rotation=45)
    plt.tick_params(axis='x', which='major', pad=-10, labelsize=8) # shift the xticks a bit to the left
    plt.xlabel("Parameters")
    plt.ylabel("Normalized values")
    plt.title("Parallel coordinates of the {} experiment".format(exp))
    plt.gca().set_facecolor((0.9, 0.9, 0.9))
    plt.axhline(y=1, color='k', linestyle='--', linewidth=0.5)
    plt.axhline(y=-1, color='k', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=0.05)
    # add legend that puts the lowest fitness value with green color and the highest with red color
    legend_elements = [Line2D([0], [0], color='r', lw=4, label='Lowest fitness: {}'.format(-fitness_range[1].round(0))),
                          Line2D([0], [0], color='g', lw=4, label='Highest fitness: {}'.format(-fitness_range[0].round(0)))]
    plt.legend(handles=legend_elements, loc='upper right')  

    plt.savefig(directory + "parallel_coordinates.pdf", format='pdf', dpi=300)


if __name__ == '__main__':
    pp = PlotParameters()
    pp.fig_height = 5
    pp.fig_width = 10
    plot_parallel_coordinates("logs/230602_bo_ls_wit_best_no_ms_ea_0.01_doe_5x_ls_30000/peg_ins/peg_ins_2/", pp)