# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMBAO)
# Code: 
# ```python
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
        for _ in range(min(self.budget, self.dim)):
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

class BlackBoxMetaheuristicOptimizer:
    def __init__(self, budget, dim, mutation_rate, exploration_rate):
        """
        Initialize the BlackBoxMetaheuristicOptimizer with a budget, dimensionality, mutation rate, and exploration rate.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
            mutation_rate (float): The probability of mutation.
            exploration_rate (float): The probability of exploration.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.population_size = 100

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxMetaheuristicOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            list: A list of optimized values.
        """
        # Initialize the population with random individuals
        population = [BlackBoxOptimizer(self.budget, self.dim).__call__(func) for _ in range(self.population_size)]

        # Evolve the population for a specified number of generations
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x[0], reverse=True)[:self.population_size // 2]

            # Perform mutation
            for individual in fittest_individuals:
                if random.random() < self.exploration_rate:
                    # Generate a random mutation point
                    point = self.search_space[np.random.randint(0, self.dim)]

                    # Perform mutation
                    individual = BlackBoxOptimizer(self.budget, self.dim).__call__(func)(point)

            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x[0], reverse=True)[:self.population_size // 2]

            # Evaluate the fittest individuals
            for individual in fittest_individuals:
                best_value = float('-inf')
                for _ in range(min(self.budget, self.dim)):
                    point = self.search_space[np.random.randint(0, self.dim)]
                    value = func(point)
                    if value > best_value:
                        best_value = value
                        break

                # Update the best individual
                individual[0] = best_value

        # Return the optimized values
        return [individual[0] for individual in population]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = BlackBoxMetaheuristicOptimizer(100, 2, 0.1, 0.01)
optimized_values = optimizer(func)
print(optimized_values)