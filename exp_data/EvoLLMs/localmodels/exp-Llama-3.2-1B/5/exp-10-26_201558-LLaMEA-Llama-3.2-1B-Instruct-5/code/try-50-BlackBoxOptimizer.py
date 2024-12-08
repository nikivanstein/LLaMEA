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

            # Apply Adaptive Perturbation to the new population
            for _ in range(self.population_size // 2):
                # Select a random individual from the new population
                individual = new_population[np.random.randint(0, self.population_size)]

                # Calculate the perturbation value
                perturbation_value = random.uniform(-1, 1)

                # Perturb the individual
                perturbed_individual = individual + perturbation_value

                # Replace the individual in the new population
                new_population[new_population == individual] = perturbed_individual

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# ```python
# ```python
# 
# def mutation(individual, budget):
#     # Select a random individual from the population
#     individual = individual[np.random.randint(0, len(individual))]

#     # Calculate the perturbation value
#     perturbation_value = random.uniform(-1, 1)

#     # Perturb the individual
#     perturbed_individual = individual + perturbation_value

#     # Replace the individual in the population
#     population[np.random.randint(0, len(individual))] = perturbed_individual

#     return perturbed_individual
# 
# def adaptive_perturbation(individual, budget):
#     # Select a random individual from the population
#     individual = individual[np.random.randint(0, len(individual))]

#     # Calculate the perturbation value
#     perturbation_value = random.uniform(-1, 1)

#     # Perturb the individual
#     perturbed_individual = individual + perturbation_value

#     # Apply Adaptive Perturbation to the individual
#     for _ in range(self.population_size // 2):
#         # Select a random individual from the new population
#         new_individual = population[np.random.randint(0, self.population_size)]

#         # Perturb the individual
#         perturbed_individual = mutation(new_individual, self.budget)

#         # Replace the individual in the new population
#         population[np.random.randint(0, self.population_size)] = perturbed_individual

#     # Replace the old population with the new population
#     population = new_population
# 
# def optimize(func, budget, dim):
#     return BlackBoxOptimizer(budget, dim), func

# ```python
# 
# # Initialize the BlackBoxOptimizer with a given budget and dimensionality
# optimizer, func = optimize("BlackBoxOptimizer", budget, dim)

# # Optimize the black box function using the BlackBoxOptimizer
# optimized_parameters, optimized_function_value = optimizer(func)

# # Print the optimized parameters and the optimized function value
# print("Optimized Parameters:", optimized_parameters)
# print("Optimized Function Value:", optimized_function_value)