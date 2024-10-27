import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    """
    An optimization algorithm that handles a wide range of tasks by leveraging the power of black box optimization.

    Attributes:
    ----------
    budget : int
        The maximum number of function evaluations allowed.
    dim : int
        The dimensionality of the search space.

    Methods:
    -------
    __init__(self, budget, dim)
        Initializes the optimization algorithm with the given budget and dimensionality.
    def __call__(self, func)
        Optimizes the black box function `func` using `self.budget` function evaluations.
    def __str__(self):
        return f"BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization"

    def __str__(self):
        return "BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization"

    def __str__(self):
        return f"BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization"