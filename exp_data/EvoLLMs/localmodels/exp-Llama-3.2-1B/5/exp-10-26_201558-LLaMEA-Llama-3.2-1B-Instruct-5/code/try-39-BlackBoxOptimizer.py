# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
import operator

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
            new_func_values = np.array([func(individual) for individual in new_population])

            # Select the fittest individuals based on the new function values
            fittest_individuals = np.argsort(new_func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

def mutation(individual, mutation_rate):
    """
    Perform a single mutation on the given individual.

    Parameters:
    individual (numpy array): The individual to mutate.
    mutation_rate (float): The probability of mutation.

    Returns:
    numpy array: The mutated individual.
    """
    # Generate a random index for mutation
    index = random.randint(0, individual.size - 1)

    # Perform the mutation
    individual[index] = random.uniform(-5.0, 5.0)

    # Return the mutated individual
    return individual

def selection(population, fitness):
    """
    Select the fittest individuals based on their fitness.

    Parameters:
    population (numpy array): The population to select from.
    fitness (numpy array): The fitness of each individual in the population.

    Returns:
    numpy array: The fittest individuals.
    """
    # Calculate the fitness of each individual
    fitness_values = np.array([fitness[i] for i in range(population.size)])

    # Select the fittest individuals based on their fitness
    fittest_individuals = np.argsort(fitness_values)[::-1][:population.size // 2]

    # Return the fittest individuals
    return fittest_individuals

def bbofbbp(budget, dim, mutation_rate, population_size, mutation_func, selection_func):
    """
    Optimize the black box function using the Evolutionary Algorithm for Black Box Optimization.

    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    mutation_rate (float): The probability of mutation.
    population_size (int): The size of the population.
    mutation_func (function): The mutation function.
    selection_func (function): The selection function.

    Returns:
    tuple: The optimized parameters and the optimized function value.
    """
    # Initialize the population with random parameters
    population = np.random.uniform(-5.0, 5.0, (population_size, dim))

    # Evaluate the function for each individual in the population
    for _ in range(budget):
        # Evaluate the function for each individual in the population
        func_values = func(population)

        # Select the fittest individuals based on the function values
        fittest_individuals = selection_func(population, func_values)

        # Create a new population by combining the fittest individuals
        new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        # Evaluate the function for each individual in the new population
        func_values = func(new_population)

        # Select the fittest individuals based on the new function values
        fittest_individuals = selection_func(new_population, func_values)

        # Perform the mutation on each individual
        new_population = np.concatenate([mutation_func(individual) for individual in fittest_individuals])

        # Replace the old population with the new population
        population = new_population

    # Return the optimized parameters and the optimized function value
    return population, func(population)

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

budget = 1000
dim = 2
mutation_rate = 0.01
population_size = 100
mutation_func = mutation
selection_func = selection

optimized_parameters, optimized_function_value = bbofbbp(budget, dim, mutation_rate, population_size, mutation_func, selection_func)

# Print the optimized parameters and the optimized function value
print("Optimized Parameters:", optimized_parameters)
print("Optimized Function Value:", optimized_function_value)

# Evaluate the function using the optimized parameters
optimized_function_value = func(optimized_parameters)
print("Optimized Function Value:", optimized_function_value)