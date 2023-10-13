from bt_learning.bo.hypermapper_integration import HypermapperOptimization

from typing import List
from behaviors.common_behaviors import ParameterizedNode


def optimize_parameters(
    param_nodes: List[ParameterizedNode],
    func,  # : Callable[[list(ParameterizedNode)], float]
    folder: str = None,  # Folder for the experiment
    iterations: int = None,  # Number of iterations
    add_priors: bool = False,  # Add priors to the parameters
    exp_name: str = None,  # Name of the experiment
    random_forest: bool = False  # Use random forest instead of gaussian process
):
    """ Optimizes the parameters of the nodes using hypermapper """
    optimizer = HypermapperOptimization(
        func=func, param_nodes=param_nodes, folder=folder, iterations=iterations, add_priors=add_priors, exp_name=exp_name, random_forest=random_forest)
    optimizer.optimize()
    return
