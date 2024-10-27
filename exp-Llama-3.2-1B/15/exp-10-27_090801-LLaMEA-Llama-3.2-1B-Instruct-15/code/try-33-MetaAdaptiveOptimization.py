# Black Box Optimization using Meta-Heuristics: "Meta-Adaptive Optimization with Adaptive Mutation and Crossover"
# Description: This algorithm combines meta-heuristics with adaptive mutation and crossover to optimize black box functions.
# Code: 
# ```python
import numpy as np
import random

class MetaAdaptiveOptimization:
    def __init__(self, budget, dim, noise_level=0.1):
        """
        Initialize the meta-adaptive optimization algorithm.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the problem.
            noise_level (float, optional): The level of noise accumulation. Defaults to 0.1.
        """
        self.budget = budget
        self.dim = dim
        self.noise_level = noise_level
        self.noise = 0
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        """
        Optimize the black box function `func` using meta-adaptive optimization.

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

            # Check if the new individual is better than the current best
            new_fitness = func(self.param_values + self.noise * np.random.normal(0, 1, self.dim))

            # Update the best individual and fitness if the new individual is better
            if new_fitness < self.best_fitness:
                self.best_individual = self.param_values
                self.best_fitness = new_fitness

        # Return the optimized parameter values and the objective function value
        return self.best_individual, self.best_fitness

    def mutate(self, individual):
        """
        Randomly mutate an individual by adding or subtracting a small random value.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Randomly select an operation (addition or subtraction) to mutate the individual
        op = random.choice(['add','subtract'])

        # Generate a random value to mutate the individual
        value = random.uniform(-5.0, 5.0)

        # Perform the mutation operation
        if op == 'add':
            individual[0] += value
        elif op =='subtract':
            individual[0] -= value

        return individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate a new child.

        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.

        Returns:
            list: The child.
        """
        # Select a random crossover point
        crossover_point = random.randint(0, self.dim - 1)

        # Split the parents into two halves
        left_half = parent1[:crossover_point]
        right_half = parent2[crossover_point:]

        # Perform crossover by combining the two halves
        child = left_half + right_half

        return child

# Initialize the meta-adaptive optimization algorithm
optimizer = MetaAdaptiveOptimization(budget=100, dim=10, noise_level=0.1)

# Optimize the black box function
individual, fitness = optimizer(__call__(lambda x: x**2))

# Print the result
print(f"Optimized individual: {individual}")
print(f"Optimized fitness: {fitness}")