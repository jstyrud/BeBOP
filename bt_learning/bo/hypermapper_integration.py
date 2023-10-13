import json
import os
from copy import deepcopy
from math import isclose
import numpy as np
from typing import List, Callable, Dict, Any
import sys
sys.path.insert(0, "./hypermapper")
from hypermapper import hypermapper  # noqa

from behaviors.common_behaviors import ParameterizedNode, ParameterTypes, NodeParameter


def replace_float32(d):
    if isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, np.float32):
                d[i] = float(v)
            elif isinstance(v, (dict, list)):
                replace_float32(v)
    elif isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, np.float32):
                d[key] = float(value)
            elif isinstance(value, (dict, list)):
                replace_float32(value)
    return d


class HypermapperOptimization:
    """Optimizes the parameters of the nodes using hypermapper"""

    def __init__(
        self,
        param_nodes: List[ParameterizedNode],
        func,  # : Callable[[list(ParameterizedNode)], float],
        iterations: int = None,
        folder: str = "/tmp",
        add_priors: bool = False,
        exp_name: str = None,
        hotstart_with_previous_parameters: bool = True,
        random_forest: bool = False,
        feasibility: bool = False,
    ):
        """Initializes the optimizer"""
        self.param_nodes = param_nodes
        self.func = func
        self.iterations = iterations
        self.folder = folder
        if self.folder[-1] != "/":
            self.folder = self.folder + "/"
        self.add_priors = add_priors
        self.random_forest = random_forest
        self.feasibility = feasibility

        # String to be added to the name of the subparameter that occure when a NodeParameter has several parameters
        self.subparam_str = "_subparam_"
        # Name for hypermapper
        if exp_name is None:
            self.application_name = "maplebenchmark"
        else:
            self.application_name = exp_name
        # Map from the name of the parameter to the NodeParameter
        self.params_map = {}
        self.params_json = {}
        # Map from the name of the parameter to the list of values
        self.param_value_list_map = {}
        self.json = {}                      # Json file to be passed to hypermapper
        self.num_params = 0
        self.resume = False
        self.num_previous_optimizations = 0
        self.new_json_filename = "scenario"
        self.last_json_filename = "scenario"
        self.last_data_file = None

        self._process_parameters()
        # Check if we resume an optimization
        self._check_existing_optimizations()
        if self.num_previous_optimizations > 0:
            self.resume = True
            if not self._check_existing_scenario(self.num_previous_optimizations):
                raise ValueError(
                    "The parameters in the scenario.json are not the same as the ones in the nodes.")
            self.application_name = self.application_name + \
                "_" + str(self.num_previous_optimizations)
            if hotstart_with_previous_parameters:
                with open(os.path.join(self.folder,  self.last_json_filename + ".json"), "r") as f:
                    scenario = json.load(f)
                    self.params_json = scenario["input_parameters"]
        self._generate_json(param_j=self.params_json)
        self._save_json_files()

    def _check_existing_scenario(self, num: int):
        # Load the existing scenario.json
        if num == 0:
            return False
        elif num > 0:
            with open(os.path.join(self.folder,  self.last_json_filename + ".json"), "r") as f:
                scenario = json.load(f)
        # check if every key in params_map appear in "input_parameters" in the scenario.json
        for key in self.params_map.keys():
            if key not in scenario["input_parameters"].keys():
                return False
        return True

    def _check_existing_optimizations(self):
        """Returns the number of optimizations already done in the folder"""
        previous_scenario_name = self.new_json_filename + ".json"
        previous_data_file = self.application_name + "_output_samples.csv"
        if previous_scenario_name in os.listdir(self.folder) and previous_data_file in os.listdir(self.folder):
            self.num_previous_optimizations = 1
            next_scenario_name = "scenario_" + \
                str(self.num_previous_optimizations) + ".json"
            next_data_file = self.application_name + "_" + \
                str(self.num_previous_optimizations) + \
                "_output_samples" + ".csv"
            while next_scenario_name in os.listdir(self.folder) and next_data_file in os.listdir(self.folder):
                self.num_previous_optimizations += 1
                previous_scenario_name = next_scenario_name
                previous_data_file = next_data_file
                next_scenario_name = "scenario_" + \
                    str(self.num_previous_optimizations) + ".json"
                next_data_file = self.application_name + "_" + \
                    str(self.num_previous_optimizations) + \
                    "_output_samples" + ".csv"
            self.new_json_filename = next_scenario_name[:-5]
            self.last_json_filename = previous_scenario_name[:-5]
            self.last_data_file = previous_data_file

    def _add_real_param_with_list(self, param_j: Dict[str, Any], name: str, np: NodeParameter, index: int = 0):
        """Add a real parameter to the json file"""
        if not isclose(np.min[index], np.max[index]):
            param_j[name] = {"parameter_type": "real",
                             "values": [np.min[index], np.max[index]]}
            if np.min[index] <= np.value[index] <= np.max[index]:
                param_j[name]["parameter_default"] = np.value[index]
            if self.add_priors and np.use_prior:
                if np.min[index] <= np.value[index] <= np.max[index]:
                    param_j[name]["prior"] = "gaussian"
                    if isinstance(np.standard_deviation, list):
                        std = np.standard_deviation[index]
                    else:
                        std = np.standard_deviation
                    param_j[name]["prior_parameters"] = [np.value[index], std]
                else:
                    raise ValueError(
                        "The value of the parameter is not within the bounds.")
            self.params_map[name] = np
            return True
        else:
            return False

    def _add_real_param(self, param_j: Dict[str, Any], name: str, np: NodeParameter):
        """Add a real parameter to the json file"""
        if not isclose(np.min, np.max):
            param_j[name] = {"parameter_type": "real",
                             "values": [np.min, np.max]}
            # if np.value is within bounds, use it as default value
            if np.min <= np.value <= np.max:
                param_j[name]["parameter_default"] = np.value
            if self.add_priors and np.use_prior:
                if np.min <= np.value <= np.max:
                    param_j[name]["prior"] = "gaussian"
                    param_j[name]["prior_parameters"] = [
                        np.value, np.standard_deviation]
                else:
                    raise ValueError(
                        "The value of the parameter is not within the bounds.")
            self.params_map[name] = np
            return True
        else:
            return False

    def _add_categorical_param(self, param_j: Dict[str, Any],  # JSON description of the parameters
                               name: str,  # Name of the new parameter
                               np: NodeParameter,  # NodeParameter reference
                               excluded_np: NodeParameter = None,   # NodeParameter that should be excluded
                               use_ordinal=True,  # Ordinals generally work better than categoricals
                               excluded_param_name: str = None  # Name of the parameter that should be excluded
                               ):
        """Add a categorical parameter to the json file"""
        if np.list_of_values is None:
            return False
        # Check if this is a learnable parameter
        new_values = []
        overlapping_values = []
        if excluded_np is None:
            new_values = np.list_of_values
        else:
            for i in range(len(np.list_of_values)):
                if np.list_of_values[i] not in excluded_np.list_of_values:
                    new_values.append(np.list_of_values[i])
                else:
                    overlapping_values.append(np.list_of_values[i])
        # Learnable
        if len(new_values) > 1 or len(overlapping_values) > 1:
            param_j[name] = dict()
            if use_ordinal:
                param_j[name]["parameter_type"] = "ordinal"
            else:
                param_j[name]["parameter_type"] = "categorical"
            # If this and the previous parameter have overlapping values, a constraint must be added
            if len(overlapping_values) > 1:
                param_j[name]["values"] = list(
                    range(len(new_values) + len(overlapping_values)))
                if excluded_param_name is None:
                    raise ValueError(
                        "excluded_param_name must be specified when there are overlapping values")
                param_j[name]["constraints"] = [
                    str(name + " != " + excluded_param_name)]
                param_j[name]["dependencies"] = [excluded_param_name]

                # Match the indices of the values that appear in both
                value_dict = {}
                for value in overlapping_values:
                    # get key value in self.param_value_list_map[excluded_param_name]
                    excluded_index = [
                        k for k, v in self.param_value_list_map[excluded_param_name].items() if v == value][0]
                    value_dict[excluded_index] = value
                # Add the new values
                i = 0
                for value in new_values:
                    while i in value_dict:
                        i += 1
                    value_dict[i] = value
                param_j[name]["values"] = list(value_dict.keys())
                self.param_value_list_map[name] = value_dict
            else:
                param_j[name]["values"] = list(range(len(new_values)))
                value_dict = {}
                for i, v in enumerate(new_values):
                    value_dict[i] = v
                self.param_value_list_map[name] = value_dict
            self.params_map[name] = np
            return True
        # Not learnable. We need to set the value to the first value in the list that is not the same as the first parameter
        elif len(new_values) == 1 and excluded_np is not None:
            for v in np.list_of_values:
                if v not in excluded_np.list_of_values:
                    np.value = v
                    break
            return False
        else:
            return False

    def _param_str(self, pn: ParameterizedNode, counter: int, subparam_counter: int = None):
        """Returns the parameter string"""
        name = "x" + str(counter)
        if pn.name is not None and pn.name != "":
            name += "_" + pn.name
        if subparam_counter is not None:
            name += self.subparam_str + str(subparam_counter)
        # remove all chraracters that are not in the regex '^[0-9a-zA-Z_-]+$'
        name = "".join(
            [c for c in name if c.isalnum() or c == "_" or c == "-"])
        return name

    def _process_parameters(self):
        """Process the parameters and generate the json file"""
        self.num_params = 0
        # Name parameters and generate parameter map
        for pn in self.param_nodes:
            if pn.parameters is not None:
                at_first_param_name = None
                for np in pn.parameters:
                    if np.data_type == ParameterTypes.FLOAT:
                        # Treat NodeParameter with floats that have multiple values as a list
                        if isinstance(np.value, list):
                            subparam_counter = 0
                            for i in range(len(np.value)):
                                if self._add_real_param_with_list(self.params_json, self._param_str(pn, self.num_params, i), np, i):
                                    subparam_counter += 1
                            if subparam_counter > 0:
                                self.num_params += subparam_counter - 1
                            else:
                                continue
                        elif not self._add_real_param(self.params_json, self._param_str(pn, self.num_params), np):
                            continue
                    elif np.data_type == ParameterTypes.POSITION:
                        subparam_counter = 0
                        for i in range(len(np.value)):
                            if self._add_real_param_with_list(self.params_json, self._param_str(pn, self.num_params, i), np, i):
                                subparam_counter += 1
                        if subparam_counter > 0:
                            self.num_params += subparam_counter - 1
                        else:
                            continue
                    elif np.list_of_values is not None:
                        # If there is nothing to learn, 'value' needs to be set to the only value in the list
                        if len(np.list_of_values) == 1:
                            np.value = np.list_of_values[0]
                            continue
                        elif len(np.list_of_values) > 1:
                            # The "at " node has a special rule that "valueA at valueA" is not allowed, so the 2nd parameter needs to handled differently
                            if pn.name == "at ":
                                if np == pn.parameters[0]:
                                    name = self._param_str(pn, self.num_params)
                                    if self._add_categorical_param(self.params_json, name, np):
                                        at_first_param_name = name
                                    else:
                                        continue
                                elif np == pn.parameters[1]:
                                    if self._add_categorical_param(self.params_json, self._param_str(pn, self.num_params), np, excluded_np=pn.parameters[0], excluded_param_name=at_first_param_name):
                                        at_first_param_name = None
                                    else:
                                        continue
                            else:
                                if not self._add_categorical_param(self.params_json, self._param_str(pn, self.num_params), np):
                                    continue
                    else:
                        continue
                    self.num_params += 1
        if self.num_params > 15:
            print("Warning: More than 15 parameters, this may take a long time to run")

    def _generate_json(self,
                       param_j: Dict[str, Any] = None):
        """Generates the json file for hypermapper"""
        j = dict()
        j["application_name"] = self.application_name
        j["run_directory"] = self.folder
        j["optimization_objectives"] = ["fitness"]
        if self.add_priors:
            j["acquisition_function"] = "EI_PIBO"
            # Weight given to the probabilistic model versus the prior in Hypermapper's posterior computation. Larger values give more emphasis to the prior. Default is 10.
            # Do not put very high values such as 100
            j["model_weight"] = 100
        if self.resume:
            j["resume_optimization"] = True
            j["resume_optimization_file"] = self.folder + self.last_data_file

        # Feasibility
        if self.feasibility:
            j["feasible_output"] = {"enable_feasible_predictor": True,
                                    "name": "feasibility", "true_value": "True", "false_value": "False"}

        # Design of experiments
        doe = {}
        doe["doe_type"] = "random sampling"
        doe["number_of_samples"] = 10 * self.num_params
        j["design_of_experiment"] = doe
        # Calculate number of iterations
        if self.iterations is not None:
            j["optimization_iterations"] = self.iterations
        else:
            j["optimization_iterations"] = 30 * \
                self.num_params - doe["number_of_samples"]

        # Parameters
        j["input_parameters"] = param_j

        # Surrogate model
        model = {}
        if self.random_forest:
            model["model"] = "random_forest"
            model["number_of_trees"] = 100  # "Number of trees in the forest." "minimum": 1, "maximum": 1000, "default": 10
            model["max_features"] = 1.0     # "Percentage of the features to be used when fitting the forest." "minimum": 0, "maximum": 1, "default": 0.5
            model["bootstrap"] = True      # "Whether to use bagging when fitting the forest." "default": false
            model["min_samples_split"] = 2  # "Minimum number of samples required to split a node.", "minimum": 2, "default": 5
            model["add_linear_std"] = True
        else:
            model["model"] = "gaussian_process"
            j["GP_model"] = "botorch"   # Available: gpy, botorch and gpytorch. We currently use botorch
        j["models"] = model
        self.json = j

        j["exploration_augmentation"] = 0.0001
        j["local_search_step_size"] = 0.1
        if not self.random_forest:
            # Set a lengthscale priors for the prior expection of the impact of each parameter
            # Other combinations could be (1.3, 0.2), (1.3, 0.3), (1.1, 0.2), (1.1, 0.3), (1.3, 0.1)
            # Recommended by Erik in May 2023: 1.2, 0.1
            j["lengthscale_prior"] = {"name": "gamma", "parameters": [1.2, 0.1]}
            # Some tuning to make it run faster
            j["local_search_improvement_threshold"] = 1e-4
            j["multistart_hyperparameter_optimization"] = False
            j["multistart_hyperparameter_optimization_initial_points"] = 5
        # New settings added in May 2023
        j["local_search_from_best"] = True
        j["local_search_random_points"] = 1000
        j["local_search_starting_points"] = 10

    def _save_json_files(self):
        """ Saves the vanilla json file. the beautified json file and the parameter mapping file"""

        self.json = replace_float32(self.json)
        with open(self.folder + self.new_json_filename + ".json", "w") as f:
            json.dump(self.json, f, indent=4)
        # Make a beautified json file and parameter mapping to file if needed
        if len(self.param_value_list_map) > 0:
            beauty_json = deepcopy(self.json)
            for k, v in beauty_json["input_parameters"].items():
                if k in self.param_value_list_map:
                    sorted_keys = sorted(self.param_value_list_map[k].keys())
                    beauty_json["input_parameters"][k]["values"] = [
                        self.param_value_list_map[k][key] for key in sorted_keys]
            with open(self.folder + self.new_json_filename + "_beautified.json", "w") as f:
                json.dump(replace_float32(beauty_json), f, indent=4)

            value_mapping_name = "value_mapping"
            if self.num_previous_optimizations > 0:
                value_mapping_name += "_" + \
                    str(self.num_previous_optimizations)
            with open(self.folder + value_mapping_name + ".json", "w") as f:
                json.dump(self.param_value_list_map, f, indent=4)

    def _evaluate_function(
        self,
        params: Dict[str, Any]
    ) -> float:
        """Evaluates the function with the given parameters"""
        for k, v in params.items():
            if self.params_map[k].data_type == ParameterTypes.FLOAT:
                # Treat NodeParameter with multiple values
                if self.subparam_str in k:
                    subparam_num = int(k.split(self.subparam_str)[1])
                    node_param = self.params_map[k]
                    node_param.value[subparam_num] = float(v)
                # Enter single value in list if there is a list, otherwise set value
                elif isinstance(self.params_map[k].value, list):
                    self.params_map[k].value[0] = float(v)
                else:
                    self.params_map[k].value = float(v)
            elif self.params_map[k].data_type == ParameterTypes.POSITION:
                # Treat NodeParameter with multiple values
                if self.subparam_str in k:
                    subparam_num = int(k.split(self.subparam_str)[1])
                    node_param = self.params_map[k]
                    value_list = list(node_param.value)
                    value_list[subparam_num] = float(v)
                    node_param.value = tuple(value_list)
                else:
                    raise ValueError(
                        "Error: Position parameters are expected to have multiple values")
            else:
                if k in self.param_value_list_map and int(v) in self.param_value_list_map[k]:
                    self.params_map[k].value = self.param_value_list_map[k][int(
                        v)]
                else:
                    raise ValueError(
                        "Error: Parameter value not found in list of values")
        # Check if an "at" node violates the rule that the first parameter cannot be the same as the second
        feasible = True
        for pn in self.param_nodes:
            if pn.name == "at " and pn.parameters[0].value == pn.parameters[1].value:
                feasible = False
                # set pn.parameters[1] to a value that is not pn.parameters[0]
                for v in pn.parameters[1].list_of_values:
                    if v != pn.parameters[0].value:
                        pn.parameters[1].value = v
                        break
                if pn.parameters[0].value == pn.parameters[1].value:
                    # set pn.parameters[0] to a value that is not pn.parameters[1]
                    for v in pn.parameters[0].list_of_values:
                        if v != pn.parameters[1].value:
                            pn.parameters[0].value = v
                            break
                if pn.parameters[0].value == pn.parameters[1].value:
                    raise ValueError(
                        "Error: Could not find a value for the second parameter of the at node that is not the same as the first")
        output = {}
        output["fitness"] = -self.func()
        if self.feasibility:
            output["feasibility"] = feasible
        return output

    def optimize(self):
        """Optimizes the parameters"""
        hypermapper.optimize(
            self.folder + self.new_json_filename + ".json", self._evaluate_function)
