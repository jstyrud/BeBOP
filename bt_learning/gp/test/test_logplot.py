"""Unit test for logplot.py module."""

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

import os
import shutil

from bt_learning.gp import logplot


DIR_NAME = os.path.abspath(os.path.dirname(__file__))


def test_moving_average():
    """ Test moving_average function """
    data = [1, 2, 3, 4, 5, 6, 7]
    ma_data = logplot.moving_average(data, 3)
    assert (ma_data == [1, 1.5, 2, 3, 4, 5, 6]).all()

    ma_data = logplot.moving_average(data, 5)
    assert (ma_data == [1, 1.5, 2, 2.5, 3, 4, 5]).all()


def test_plot_fitness():
    """Test plot_fitness function."""
    logplot.clear_logs('tests/test')
    logplot.plot_fitness('tests/test', [0, 1, 2])
    assert os.path.isfile(logplot.get_log_folder('tests/test') + '/Fitness_Episodes.png')
    try:
        shutil.rmtree(logplot.get_log_folder('tests/test'))
    except FileNotFoundError:
        pass


def test_extend_gens():
    """Test plot of learning curves with extended gens."""
    file_name = os.path.join(DIR_NAME, 'test.pdf')
    logplot.clear_logs('tests/test1')
    logplot.log_best_net_fitness('tests/test1', [1.0, 2.0, 3.0, 4.0, 5.0])
    logplot.log_n_episodes('tests/test1', [5, 10, 15, 20, 25])
    logplot.clear_logs('tests/test2')
    logplot.log_best_net_fitness('tests/test2', [1, 2, 5])
    logplot.log_n_episodes('tests/test2', [5.0, 10.0, 15.0])
    parameters = logplot.PlotParameters()
    parameters.path = file_name
    parameters.extend_gens = 5
    parameters.save_fig = True
    parameters.x_max = 30
    logplot.plot_learning_curves(['tests/test1', 'tests/test2'], parameters)


def test_plot_learning_curves():
    """Test plot_learning_curves function."""
    file_name = os.path.join(DIR_NAME, 'test.pdf')
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass

    logplot.clear_logs('tests/test')
    logplot.log_best_net_fitness('tests/test', [1.0, 2.0, 3.0, 4.0, 5.0])
    logplot.log_n_episodes('tests/test', [5, 10, 15, 20, 25])

    parameters = logplot.PlotParameters()
    parameters.path = file_name
    parameters.extrapolate_y = False
    parameters.plot_mean = False
    parameters.plot_std = False
    parameters.plot_ind = False
    parameters.save_fig = False
    parameters.x_max = 0
    parameters.plot_horizontal = True
    logplot.plot_learning_curves(['tests/test'], parameters)
    assert not os.path.isfile(file_name)

    parameters.extrapolate_y = True
    parameters.plot_mean = True
    parameters.plot_std = True
    parameters.plot_ind = True
    parameters.save_fig = True
    parameters.plot_minmax = True
    parameters.x_max = 100
    parameters.y_max = 100
    parameters.y_min = 10
    parameters.logarithmic_y = True
    parameters.plot_horizontal = True
    parameters.save_fig = True
    logplot.plot_learning_curves(['tests/test'], parameters)
    assert os.path.isfile(file_name)
    os.remove(file_name)

    parameters.x_max = 10
    parameters.plot_horizontal = False
    logplot.plot_learning_curves(['tests/test'], parameters)
    assert os.path.isfile(file_name)

    os.remove(file_name)
    try:
        shutil.rmtree(logplot.get_log_folder('tests/test'))
    except FileNotFoundError:
        pass


def test_best_individual():
    """Test logging and loading a best individual."""
    best_individual = ['a']
    logplot.clear_logs('tests/test')
    logplot.log_best_individual('tests/test', best_individual)

    loaded_best_individual = logplot.get_best_individual('tests/test')
    assert best_individual == loaded_best_individual
