import numpy as np
import random
import os

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

    def mutate(self):
        """
        Mutate the selected solution to refine its strategy.

        Returns:
            tuple: A tuple containing the mutated parameter values and the objective function value.
        """
        # Select a random individual from the population
        individual = np.random.choice(self.population, self.dim, replace=False)

        # Apply a mutation to the individual
        mutated_individual = individual + random.uniform(-1, 1) * np.random.normal(0, 1, self.dim)

        # Evaluate the mutated individual
        mutated_func_value = func(mutated_individual)

        # Return the mutated individual and its objective function value
        return mutated_individual, mutated_func_value

# Define a fitness function to evaluate the objective function
def fitness(individual):
    """
    Evaluate the objective function using the given individual.

    Args:
        individual (numpy array): The individual to evaluate.

    Returns:
        float: The objective function value.
    """
    func_value = func(individual)
    return func_value

# Define the BBOB test suite
class BBOB:
    def __init__(self, func, noise_level=0.1):
        """
        Initialize the BBOB test suite.

        Args:
            func (callable): The black box function to optimize.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.func = func
        self.noise_level = noise_level

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of the given individual.

        Args:
            individual (numpy array): The individual to evaluate.

        Returns:
            float: The fitness value.
        """
        func_value = fitness(individual)
        return func_value

# Define the BBOB test suite with 24 noiseless functions
class BBOB_Noiseless:
    def __init__(self):
        """
        Initialize the BBOB test suite with 24 noiseless functions.
        """
        self.funcs = [lambda x: x**2, lambda x: np.sin(x), lambda x: x**3, lambda x: x**4, lambda x: x**5, lambda x: x**6, lambda x: x**7, lambda x: x**8, lambda x: x**9, lambda x: x**10, lambda x: x**11, lambda x: x**12, lambda x: x**13, lambda x: x**14, lambda x: x**15, lambda x: x**16]
        self.noise_level = 0.1

# Initialize the meta-gradient descent algorithm
meta_gradient_descent = MetaGradientDescent(1000, 10)

# Initialize the BBOB test suite
bbob = BBOB_Noiseless()

# Evaluate the objective function for the first 1000 iterations
for _ in range(1000):
    meta_gradient_descent(bbob.evaluate_fitness(meta_gradient_descent.param_values))

# Mutate the selected solution and evaluate the objective function
for _ in range(100):
    mutated_individual, mutated_func_value = meta_gradient_descent.mutate()
    bbob.evaluate_fitness(mutated_individual)

# Update the meta-gradient descent algorithm
meta_gradient_descent.param_values = bbob.func(mutated_individual)

# Print the final solution
print("Final Solution:", meta_gradient_descent.param_values)

# Save the final solution to a file
np.save("final_solution.npy", meta_gradient_descent.param_values)