import numpy as np
import random
from scipy.optimize import minimize

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

    def mutate(self, individual):
        """
        Mutate the individual to refine its strategy.

        Args:
            individual (numpy array): The current individual.

        Returns:
            numpy array: The mutated individual.
        """
        # Randomly select a mutation point within the search space
        mutation_point = random.randint(0, self.dim - 1)

        # Swap the mutation point with a random individual from the search space
        mutated_individual = np.copy(individual)
        mutated_individual[mutation_point], mutated_individual[random.randint(0, self.dim - 1)] = mutated_individual[random.randint(0, self.dim - 1)], mutated_individual[mutation_point]

        return mutated_individual

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of the individual.

        Args:
            individual (numpy array): The individual to evaluate.

        Returns:
            float: The fitness value of the individual.
        """
        # Evaluate the objective function with the accumulated noise
        func_value = func(individual + self.noise * np.random.normal(0, 1, self.dim))

        # Update the parameter values based on the accumulated noise
        self.param_values = individual + self.noise * np.random.normal(0, 1, self.dim)

        return func_value

def func(individual):
    """
    The black box function to optimize.

    Args:
        individual (numpy array): The individual to optimize.

    Returns:
        float: The objective function value.
    """
    # Simulate a noisy function evaluation
    return np.sin(individual[0]) + np.sin(individual[1])

# Initialize the meta-gradient descent algorithm
meta_grad_descent = MetaGradientDescent(budget=100, dim=10, noise_level=0.1)

# Optimize the BBOB function using meta-gradient descent
optimized_individual = meta_grad_descent(func)

# Print the optimized individual and its fitness
print("Optimized Individual:", optimized_individual)
print("Fitness:", func(optimized_individual))

# Mutate the optimized individual to refine its strategy
mutated_individual = meta_grad_descent.mutate(optimized_individual)

# Print the mutated individual and its fitness
print("Mutated Individual:", mutated_individual)
print("Fitness:", func(mutated_individual))

# Evaluate the fitness of the mutated individual
mutated_fitness = meta_grad_descent.evaluate_fitness(mutated_individual)

# Print the fitness of the mutated individual
print("Fitness of Mutated Individual:", mutated_fitness)

# Update the meta-gradient descent algorithm with the mutated individual
meta_grad_descent.param_values = mutated_individual

# Optimize the BBOB function using the updated meta-gradient descent algorithm
optimized_mutated_individual = meta_grad_descent(func)

# Print the optimized mutated individual and its fitness
print("Optimized Mutated Individual:", optimized_mutated_individual)
print("Fitness:", func(optimized_mutated_individual))

# Print the fitness of the optimized mutated individual
print("Fitness of Optimized Mutated Individual:", func(optimized_mutated_individual))