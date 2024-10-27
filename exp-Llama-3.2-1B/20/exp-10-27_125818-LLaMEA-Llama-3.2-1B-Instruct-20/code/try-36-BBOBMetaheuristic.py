# Description: Black Box Optimization using Evolutionary Algorithm
# Code: 
# ```python
import numpy as np
import random
import copy

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        """
        Initialize the BBOBMetaheuristic with a given budget and dimensionality.

        Args:
        - budget: The maximum number of function evaluations allowed.
        - dim: The dimensionality of the optimization problem.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.space = None
        self.x = None
        self.f = None
        self.population = None
        self.population_history = None
        self.fitness_history = None

    def __call__(self, func):
        """
        Optimize the black box function `func` using `self.budget` function evaluations.

        Args:
        - func: The black box function to be optimized.

        Returns:
        - The optimized function value.
        """
        if self.func is None:
            self.func = func
            self.space = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.x = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.f = self.func(self.x)
            self.population = [self.x.copy() for _ in range(100)]
            self.population_history = [self.f.copy() for _ in range(100)]
            self.fitness_history = [self.f.copy() for _ in range(100)]
        else:
            while self.budget > 0:
                # Sample a new point in the search space
                new_individual = copy.deepcopy(self.x)
                # Evaluate the function at the new point
                new_fitness = self.func(new_individual)
                # Check if the new point is better than the current point
                if new_fitness < self.f + 1e-6:  # add a small value to avoid division by zero
                    # Update the current point
                    new_individual = copy.deepcopy(self.x)
                    new_individual[0] += random.uniform(-0.1, 0.1)
                    new_individual[1] += random.uniform(-0.1, 0.1)
                    self.x = new_individual
                    self.f = new_fitness
                    self.population.append(new_individual)
                    self.population_history.append(self.f)
                    self.fitness_history.append(self.f)
                else:
                    # Select the fittest individual
                    self.population.sort(key=lambda individual: self.fitness_history[-1], reverse=True)
                    self.x = self.population[0]
                    self.f = self.population_history[0]
                    self.population = [self.x.copy() for _ in range(100)]
                    self.population_history = [self.f.copy() for _ in range(100)]
                    self.fitness_history = [self.f.copy() for _ in range(100)]
                # Decrease the budget
                self.budget -= 1
            # Return the optimized function value
            return self.f

# Description: Black Box Optimization using Evolutionary Algorithm
# Code: 
# ```python
bboo_metaheuristic = BBOBMetaheuristic(1000, 5)