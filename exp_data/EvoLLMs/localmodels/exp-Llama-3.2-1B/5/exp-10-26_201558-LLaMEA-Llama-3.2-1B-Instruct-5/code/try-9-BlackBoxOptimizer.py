# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import minimize

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

            # Evaluate the function for each individual in the new population
            func_values = func(new_population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

def mutation(individual, func, budget):
    """
    Apply a novel mutation strategy to the given individual.

    Parameters:
    individual (numpy array): The individual to mutate.
    func (function): The black box function.
    budget (int): The maximum number of function evaluations allowed.

    Returns:
    tuple: The mutated individual and the mutated function value.
    """
    # Evaluate the function for the individual
    func_value = func(individual)

    # Select a random function evaluation
    evaluation = random.randint(0, budget - 1)

    # Apply the mutation
    if evaluation < len(individual):
        individual[evaluation] += random.uniform(-1, 1)

    # Return the mutated individual and the mutated function value
    return individual, func_value

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# BlackBoxOptimizer: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# 
# Parameters:
# budget (int): The maximum number of function evaluations allowed.
# dim (int): The dimensionality of the search space.
# 
# Returns:
# tuple: The optimized parameters and the optimized function value.
# ```
# ```python
# ```python
optimizer = BlackBoxOptimizer(budget=100, dim=10)

# Optimize the black box function
optimized_parameters, optimized_function_value = optimizer(func, np.linspace(-5.0, 5.0, 100))

# Print the optimized parameters and the optimized function value
print("Optimized Parameters:", optimized_parameters)
print("Optimized Function Value:", optimized_function_value)