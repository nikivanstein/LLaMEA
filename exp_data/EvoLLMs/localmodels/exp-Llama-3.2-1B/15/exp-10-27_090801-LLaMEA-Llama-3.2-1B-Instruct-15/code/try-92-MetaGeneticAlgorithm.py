import numpy as np
import random
import os

class MetaGeneticAlgorithm:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-genetic algorithm.

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
        Optimize the black box function `func` using meta-genetic algorithm.

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
        for i in range(self.budget):
            # Update the objective function value
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the probability of changing the individual lines
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Update the objective function value
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the parameter values based on the probability of changing the individual lines
            self.param_values += self.noise * np.random.normal(0, 1, self.dim)

            # Update the objective function value
            func_value = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

        # Return the optimized parameter values and the objective function value
        return self.param_values, func_value

# Example usage:
if __name__ == "__main__":
    # Initialize the meta-genetic algorithm with a budget of 1000 evaluations
    meta_genetic_algorithm = MetaGeneticAlgorithm(1000, 10)

    # Optimize the black box function using the meta-genetic algorithm
    func = lambda x: x**2
    optimized_params, optimized_func_value = meta_genetic_algorithm(func)

    # Save the optimized solution to a file
    os.makedirs("optimized_solutions", exist_ok=True)
    np.save("optimized_solutions/optimized_params.npy", optimized_params)
    np.save("optimized_solutions/optimized_func_value.npy", optimized_func_value)