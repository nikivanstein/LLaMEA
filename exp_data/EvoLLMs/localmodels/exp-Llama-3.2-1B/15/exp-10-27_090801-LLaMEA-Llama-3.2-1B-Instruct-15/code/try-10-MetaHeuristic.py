import numpy as np
import random

class MetaHeuristic:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-heuristic algorithm.

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
        Optimize the black box function `func` using meta-heuristic.

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

        # Refine the solution by changing the individual lines of the selected solution
        self.param_values = np.clip(self.param_values, -5.0, 5.0)

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

    def __str__(self):
        """
        Return a string representation of the meta-heuristic algorithm.

        Returns:
            str: A string containing the algorithm's name, description, and score.
        """
        return "MetaHeuristic: Optimize black box function using evolutionary strategies\n" \
               "Description: A novel metaheuristic algorithm for solving black box optimization problems\n" \
               f"Code: {self.__class__.__name__}\n" \
               f"Score: -inf"

# Example usage:
metaheuristic = MetaHeuristic(100, 10)  # Initialize the meta-heuristic with a budget of 100 and dimensionality of 10
func = lambda x: x**2  # Define a simple black box function
optimized_solution = metaheuristic(func)  # Optimize the function using the meta-heuristic
print(optimized_solution)