import json
import math
import numpy as np
from typing import List

from behaviors.common_behaviors import ParameterizedNode, ParameterTypes, NodeParameter
from bt_learning.bo.bo import optimize_parameters

### Helpers

def get_float(dim=1, min=-1., max=1.):
    return NodeParameter(value=[0.]*dim, min=[min]*dim, max=[max]*dim, data_type=ParameterTypes.FLOAT, standard_deviation=1.)

def get_node_param_objects(
        param_nodes: List[ParameterizedNode]) -> List[NodeParameter]:
    """Returns a list of the parameters of the nodes"""
    bt_params = []
    for param_node in param_nodes:
        if param_node.parameters:
            for param in param_node.parameters:
                if param.data_type == ParameterTypes.FLOAT:
                    if isinstance(param.value, list):
                        for i in range(len(param.value)):
                            if not math.isclose(param.min[i], param.max[i]):
                                bt_params.append(param)
                                break
                    else:
                        if not math.isclose(param.min, param.max):
                            bt_params.append(param)
                elif param.list_of_values is not None and len(param.list_of_values) > 1:
                    bt_params.append(param)
    return bt_params

def get_node_parameters_values(
    parameters: List[NodeParameter]
) -> np.ndarray:
    """Returns a list of the values of the parameters"""
    b = []
    for p in parameters:
        if isinstance(p.value, list):
            b.extend(p.value)
        else:
            b.append(p.value)
    return np.asarray(b)

### Functions

class branin2:
    def __init__(self, param_nodes: List[ParameterizedNode]):
        self.param_nodes = param_nodes

    def branin2_function(self, X):
        x1 = X['x1']
        x2 = X['x2']
        a = 1.0
        b = 5.1 / (4.0 * math.pi * math.pi)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)
        y_value = a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + \
            s * (1 - t) * math.cos(x1) + s
        return y_value

    def f(self):
        X = get_node_parameters_values(get_node_param_objects(self.param_nodes))
        return self.branin2_function({'x1': X[0], 'x2': X[1]})


def get_single_param_branin2_nodes():
    a = ParameterizedNode('a', None, [get_float(dim=1, min=-5, max=10)], True)
    b = ParameterizedNode('b', None, [get_float(dim=1, min=0, max=15)], True)
    return [a, b]


def get_double_param_branin2_node(violate_bounds=False):
    a = ParameterizedNode('a', None, [NodeParameter(value=[
                          0., 0.], min=[-5., 0.], max=[10., 15.], data_type=ParameterTypes.FLOAT)], True)
    if violate_bounds:
        a.parameters[0].value = [-10., 0.]
    return [a]


def get_single_param_branin2_node():
    a = ParameterizedNode('a', None, [NodeParameter(value=0., min=-5., max=10., data_type=ParameterTypes.FLOAT),
                          NodeParameter(value=[0.], min=[0.], max=[15.], data_type=ParameterTypes.FLOAT)], True)
    return [a]

### JSON Checks

def load_json(filename: str):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'File {filename} not found')


def num_input_parameters(j : dict):
    return len(j['input_parameters'])

def num_parameters_by_type(j : dict, parameter_type : str):
    if j["input_parameters"] is None:
        return 0
    num = 0
    for pv in j['input_parameters'].values():
        if pv['parameter_type'] == parameter_type:
            num += 1
    return num


###  Actual tests
def test_bo_branin2_single():
    b = branin2(get_single_param_branin2_nodes())

    optimize_parameters(
        param_nodes=b.param_nodes, func=b.f, iterations=7, add_priors=False, folder='/tmp')
    j = load_json('/tmp/scenario.json')
    assert num_input_parameters(j) == 2
    assert num_parameters_by_type(j, 'real') == 2

    b = branin2(get_single_param_branin2_nodes())
    optimize_parameters(
        param_nodes=b.param_nodes, func=b.f, iterations=7, add_priors=True, folder='/tmp')
    j = load_json('/tmp/scenario.json')
    assert num_input_parameters(j) == 2
    assert num_parameters_by_type(j, 'real') == 2


def test_bo_branin2_double():
    b = branin2(get_double_param_branin2_node())
    optimize_parameters(
        param_nodes=b.param_nodes, func=b.f, iterations=7)
    j = load_json('/tmp/scenario.json')
    assert num_input_parameters(j) == 2
    assert num_parameters_by_type(j, 'real') == 2

    
def test_bo_branin2_double_violate_bounds():
    b = branin2(get_double_param_branin2_node(violate_bounds=True))
    optimize_parameters(
        param_nodes=b.param_nodes, func=b.f, iterations=7, add_priors=False)

    b = branin2(get_double_param_branin2_node(violate_bounds=True))
    try:
        optimize_parameters(
            param_nodes=b.param_nodes, func=b.f, iterations=7, add_priors=True)
        raise Exception("Should have failed")
    except ValueError:
        pass


def test_bo_branin2_single_double():
    b = branin2(get_single_param_branin2_nodes())
    optimize_parameters(
        param_nodes=b.param_nodes, func=b.f, iterations=7)
    j = load_json('/tmp/scenario.json')
    assert num_input_parameters(j) == 2
    assert num_parameters_by_type(j, 'real') == 2


if __name__ == '__main__':
    test_bo_branin2_single()
    test_bo_branin2_double()
    test_bo_branin2_single_double()
    test_bo_branin2_double_violate_bounds()
