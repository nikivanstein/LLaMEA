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
            for i in range(population_size):
                if random.random() < 0.05:
                    new_population[i] = func_values[i] + random.uniform(-0.1, 0.1)

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

def fitness(individual, func):
    """
    Evaluate the fitness of an individual in the population.

    Parameters:
    individual (numpy array): The individual to evaluate.
    func (function): The black box function to evaluate.

    Returns:
    float: The fitness value of the individual.
    """
    func_values = func(individual)
    return np.mean(func_values)

# Example usage:
def black_box_func(x):
    return x**2 + 2*x + 1

optimizer = BlackBoxOptimizer(budget=100, dim=2)
individual = np.random.uniform(-5.0, 5.0, (1, 2))
optimized_individual, optimized_func = optimizer(individual)
fitness_value = fitness(optimized_individual, black_box_func)
print("Optimized individual:", optimized_individual)
print("Optimized function value:", optimized_func)
print("Fitness value:", fitness_value)