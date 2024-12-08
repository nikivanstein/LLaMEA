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

            # Select the fittest individuals to refine the strategy
            fittest_individuals = np.random.choice(fittest_individuals, self.population_size, replace=False)

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

def mutate(individual, mutation_rate):
    """
    Apply a novel mutation strategy to the given individual.

    Parameters:
    individual (numpy array): The individual to mutate.
    mutation_rate (float): The probability of applying a mutation.

    Returns:
    numpy array: The mutated individual.
    """
    if random.random() < mutation_rate:
        # Select a random dimension and apply a mutation
        dim = individual.shape[1]
        index = random.randint(0, dim - 1)
        individual[index] += random.uniform(-1, 1)
        individual[index] = max(-5.0, min(individual[index], 5.0))
    return individual

# Test the algorithm
budget = 1000
dim = 5
optimizer = BlackBoxOptimizer(budget, dim)

# Evaluate the function 1000 times
func_values = []
for _ in range(budget):
    func_values.append(func(np.random.uniform(-5.0, 5.0, dim)))

# Optimize the function
optimized_individuals, optimized_function_values = optimizer(__call__, func)

# Print the results
print("Optimized Individuals:")
for i, individual in enumerate(optimized_individuals):
    print(f"Individual {i+1}: {individual}")
print(f"Optimized Function Values: {optimized_function_values}")