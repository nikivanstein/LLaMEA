# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
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

            # Apply the mutation strategy
            for i in range(new_population.shape[0]):
                if random.random() < 0.05:  # 5% chance of mutation
                    new_individual = new_population[i]
                    new_individual[0] = random.uniform(-5.0, 5.0)  # Randomly change the lower bound
                    new_individual[1] = random.uniform(-5.0, 5.0)  # Randomly change the upper bound
                    new_individual = self.evaluate_fitness(new_individual, func)  # Evaluate the new individual

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual in the population.

        Parameters:
        individual (numpy array): The individual to evaluate.
        func (function): The black box function to evaluate.

        Returns:
        float: The fitness of the individual.
        """
        func_values = func(individual)
        return np.mean(func_values)

# Example usage:
if __name__ == "__main__":
    # Create an instance of the BlackBoxOptimizer
    optimizer = BlackBoxOptimizer(budget=100, dim=10)

    # Optimize the black box function
    optimized_individual, optimized_function = optimizer(func, x)

    # Print the optimized parameters and function value
    print("Optimized Parameters:", optimized_individual)
    print("Optimized Function Value:", optimized_function)