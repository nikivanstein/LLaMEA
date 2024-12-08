import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Refine the strategy with a novel mutation
            if random.random() < 0.05:
                # Randomly select an individual from the population
                individual = population[np.random.choice(population_size)]

                # Apply a mutation to the selected individual
                mutated_individual = individual + random.uniform(-0.1, 0.1)

                # Ensure the mutated individual is within the search space
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

                # Replace the mutated individual in the population
                population[np.random.choice(population_size), :] = mutated_individual

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Example usage:
def func(x):
    return np.sum(x**2)

optimizer = BlackBoxOptimizer(budget=1000, dim=10)
optimized_params, optimized_func = optimizer(func)

# Print the optimized parameters and function value
print("Optimized Parameters:", optimized_params)
print("Optimized Function Value:", optimized_func)