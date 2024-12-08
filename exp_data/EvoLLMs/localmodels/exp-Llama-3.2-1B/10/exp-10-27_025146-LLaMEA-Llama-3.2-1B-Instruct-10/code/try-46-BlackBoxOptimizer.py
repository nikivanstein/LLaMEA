import random
import numpy as np

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

    def __call__(self, func, budget=100):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            budget (int, optional): The maximum number of function evaluations allowed. Defaults to 100.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(budget):
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

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code:
# ```python
def novel_metaheuristic(budget=100, dim=5, initial_population=100, mutation_rate=0.01):
    """
    Novel Metaheuristic Algorithm for Black Box Optimization.

    Args:
        budget (int, optional): The maximum number of function evaluations allowed. Defaults to 100.
        dim (int, optional): The dimensionality of the search space. Defaults to 5.
        initial_population (int, optional): The initial population size. Defaults to 100.
        mutation_rate (float, optional): The mutation rate. Defaults to 0.01.

    Returns:
        list: The optimized population.
    """
    # Initialize the population with random initial individuals
    population = [BlackBoxOptimizer(budget, dim) for _ in range(initial_population)]

    # Evolve the population over generations
    for _ in range(100):
        # Select the fittest individuals
        fittest_individuals = sorted(population, key=lambda x: x.budget, reverse=True)[:initial_population]

        # Perform crossover and mutation
        offspring = []
        for _ in range(len(fittest_individuals)):
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = parent1.__call__(func, budget) + mutation_rate * (parent2.budget - parent1.budget)
            offspring.append(child)

        # Replace the least fit individuals with the new offspring
        population = [child for child in offspring if child.budget > population[0].budget] + [child for child in offspring if child.budget <= population[0].budget]

    # Return the optimized population
    return population

# Example usage:
func = lambda x: x**2
budget = 100
dim = 5
optimized_population = novel_metaheuristic(budget, dim, mutation_rate=0.01)
print(optimized_population[0].budget)