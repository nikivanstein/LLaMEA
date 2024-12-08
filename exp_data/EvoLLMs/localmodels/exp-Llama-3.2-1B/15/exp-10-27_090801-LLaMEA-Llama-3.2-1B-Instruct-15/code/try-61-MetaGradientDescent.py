import numpy as np
import random

class MetaGradientDescent:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-gradient descent algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-gradient descent.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized parameter values and the objective function value.
        """
        # Initialize the parameter values to random values within the search space
        self.param_values = np.random.uniform(-5.0, 5.0, self.dim)

        # Accumulate noise in the objective function evaluations
        for _ in range(self.budget):
            # Evaluate the objective function with accumulated noise
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the accumulated noise
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def select_strategy(self):
        """
        Select a strategy for refining the solution.

        Returns:
            tuple: A tuple containing the updated parameter values and the objective function value.
        """
        # Select a random strategy from the list of possible strategies
        strategies = [
            {"name": "line_search", "func": self.line_search},
            {"name": "bounded_line_search", "func": self.bounded_line_search},
            {"name": "random_search", "func": self.random_search},
            {"name": "bounded_random_search", "func": self.bounded_random_search},
        ]

        # Select a random strategy from the list of possible strategies
        selected_strategy = random.choice(strategies)

        # Refine the solution based on the selected strategy
        return self.select_strategy_from_strategy(selected_strategy)

    def select_strategy_from_strategy(self, selected_strategy):
        """
        Select a strategy from a list of possible strategies.

        Args:
            selected_strategy (dict): A dictionary containing the selected strategy.

        Returns:
            tuple: A tuple containing the updated parameter values and the objective function value.
        """
        # Select a random strategy from the list of possible strategies
        strategies = {
            "line_search": lambda x: self.line_search(x),
            "bounded_line_search": lambda x: self.bounded_line_search(x),
            "random_search": lambda x: self.random_search(x),
            "bounded_random_search": lambda x: self.bounded_random_search(x),
        }

        # Select a random strategy from the list of possible strategies
        strategy = random.choice(list(strategies.keys()))

        # Refine the solution based on the selected strategy
        return strategies[strategy](x=self.param_values, func=selected_strategy["func"])

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 