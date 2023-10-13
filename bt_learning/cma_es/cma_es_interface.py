""" An interface to cma-es """

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

import cma
import os
import pickle


def normalize(value, high, low):
    """ Input an unnormalized value and output normalized between -1 and 1 in the working range """
    return (2.0 * value - high - low) / (high - low)


def denormalize(value, high, low):
    """ Input an normalized value between -1 and 1 and output denormalized value in the working range """
    return (value * (high - low) + high + low) / 2.0


def optimize_parameters(parameterized_nodes, fitness_function, log_folder, iterations):
    """ Runs cma-es to optimize parameters of input nodes """
    if log_folder[-1] != '/':
        log_folder += '/'
    start_values = []
    for node in parameterized_nodes:
        if node.parameters is not None:
            for parameter in node.parameters:
                if parameter.max != parameter.min:
                    if isinstance(parameter.value, tuple):
                        for i, value in enumerate(parameter.value):
                            if parameter.max[i] != parameter.min[i]:
                                start_values.append(normalize(value, parameter.max[i], parameter.min[i]))
                    elif isinstance(parameter.value, float):
                        start_values.append(normalize(parameter.value, parameter.max, parameter.min))

    # TODO better stop/restart criteria, don't need to optimize so closely
    # TODO popsize seems to work fine with default, is it ok for all benchmarks?
    # defaults to '4 + 3 * np.log(N)
    # TODO there could be some way to set number of evaluations in CMA-ES? (NoiseHandler),
    # Test if better than our 80% solutions
    if iterations is None:
        iterations = 20 * len(start_values)

    sigma = 0.5
    # Check if es.pickle exists and load it if it's there
    if os.path.exists(log_folder + "es.pickle"):
        es = pickle.load(open(log_folder + "es.pickle", 'rb'))
        if "maxfevals" in es._stopdict:
            es.opts["maxfevals"] = iterations
        # TODO: It does not append the existing files
        logger = cma.CMADataLogger(log_folder).load()
        logger.register(es)
    else:
        opts = cma.CMAOptions()
        opts.set('bounds', [[-1], [1]])
        opts.set('maxfevals', iterations)
        opts.set('tolfun', 1.0)  #TODO, this checks also history, should we do our own? History this far back: 10 + 30 * N / sp.popsize:
        opts.set('tolfunrel', 0.01)
        opts.set('tolx', 0.01)
        opts.set('tolstagnation', 50)  # TODO This takes at least N * (5 + 100 / es.popsize) iterations anyway, do our own?
        #opts.set('CMA_elitist', True)

        # Interesting parameters:
        # "popsize": "4 + 3 * np.log(N)  # population size, AKA lambda, int(popsize) is the number of new solution per iteration",
        # Some abort criteria:
        # "tolflatfitness": "1  #v iterations tolerated with flat fitness before termination",
        # "tolfun": "1e-11  #v termination criterion: tolerance in function value, quite useful",
        # "tolfunhist": "1e-12  #v termination criterion: tolerance in function value history",
        # "tolfunrel": "0  #v termination criterion: relative tolerance in function value: Delta f current < tolfunrel * (median0 - median_min)",
        # "tolstagnation": "int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations",
        # "tolx": "1e-11  #v termination criterion: tolerance in x-changes",

        es = cma.CMAEvolutionStrategy(start_values, sigma, opts)
        logger = cma.CMADataLogger(log_folder).register(es)

    best_solution = []
    best_fitness = 9999999999999999999

    def objective_function(parameters):
        parameter_index = 0
        for node in parameterized_nodes:
            if node.parameters is not None:
                for parameter in node.parameters:
                    if parameter.max != parameter.min:
                        if isinstance(parameter.value, tuple):
                            values = list(parameter.value)
                            for i in range(len(parameter.value)):
                                if parameter.max[i] != parameter.min[i]:
                                    values[i] = denormalize(parameters[parameter_index], parameter.max[i], parameter.min[i])
                                    parameter_index += 1
                            parameter.value = tuple(values)
                        elif isinstance(parameter.value, float):
                            parameter.value = denormalize(parameters[parameter_index], parameter.max, parameter.min)
                            parameter_index += 1
        return -1.0 * fitness_function()  # CMA-ES is a minimizer

    # TODO figure out is setting n_jobs is beneficial, n_jobs (-1 = all cpu's, default just one) maybe like 6?
    save_best = False
    if save_best:
        while not es.stop(check_in_same_iteration=True):
            X = es.ask()  # deliver candidate solutions
            fitnesses = []

            for x in X:
                if best_solution != [] and (x == best_solution).all():
                    fitness = best_fitness
                else:
                    fitness = objective_function(x)
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_solution = x
                fitnesses.append(fitness)
            es.tell(X, fitnesses)  # all the work is done here
            es.inject([best_solution], force=True)  # Make sure best solution is always in the solution set
            es.disp(10)  # disp does nothing if not overwritten
            logger.add()
    else:
        es.optimize(objective_function, maxfun=iterations, verb_disp=10)
    es.result_pretty()
    #logger.plot_all()
    with open(log_folder + "es.pickle", 'wb') as f:
        f.write(es.pickle_dumps())
