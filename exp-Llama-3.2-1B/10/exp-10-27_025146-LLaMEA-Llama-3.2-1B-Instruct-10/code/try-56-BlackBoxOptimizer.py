# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

def black_box_optimize(func, budget, dim, mutation_rate):
    """
    Optimize a black box function using the Novel Metaheuristic Algorithm for Black Box Optimization.

    Args:
        func (callable): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        mutation_rate (float): The rate of mutation in the population.

    Returns:
        float: The optimized value of the function.
    """
    # Initialize the population with random individuals
    population = np.random.uniform(-5.0, 5.0, (budget, dim))

    # Evolve the population for the specified number of generations
    for _ in range(100):
        # Evaluate the fitness of each individual
        fitness = np.array([func(individual) for individual in population])

        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[::-1][:budget // 2]

        # Create a new population by mutating the fittest individuals
        new_population = np.copy(population)
        for i in range(budget // 2):
            # Select two individuals from the fittest individuals
            parent1, parent2 = fittest_individuals[i], fittest_individuals[i + 1]

            # Perform mutation
            if random.random() < mutation_rate:
                # Swap the two individuals
                new_population[i, np.random.randint(0, dim)] = parent2
                new_population[i + 1, np.random.randint(0, dim)] = parent1

        # Replace the old population with the new one
        population = new_population

    # Evaluate the fitness of the final population
    fitness = np.array([func(individual) for individual in population])

    # Return the fittest individual
    return np.argmax(fitness)

# Example usage
budget = 100
dim = 5
mutation_rate = 0.01
best_individual = black_box_optimize(lambda x: x[0] + x[1], budget, dim, mutation_rate)
print(f"Best individual: {best_individual}")