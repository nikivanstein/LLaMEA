import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_individual = None
        self.best_value = -np.inf
        self.iterations = 0

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a new population
            self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            # Evaluate the function at each individual in the new population
            self.func_evals = 0
            for individual in self.population:
                value = func(individual)
                # Check if the individual is within the bounds
                if -5.0 <= individual[0] <= 5.0 and -5.0 <= individual[1] <= 5.0:
                    # If the individual is within bounds, update the function value
                    self.func_evals += 1
                    self.population[self.iterations, :] = individual
                    self.best_individual = individual
                    self.best_value = max(self.best_value, value)
                    # Refine the strategy by changing the probability of mutation
                    if np.random.rand() < 0.45:
                        self.population[self.iterations, :] = np.random.uniform(-5.0, 5.0, self.dim)
            # Select the best individual
            if self.func_evals == self.budget:
                self.best_individual = self.population[np.argmax([self.best_value, np.max(self.func(self.population[:, :]))])]
            self.iterations += 1
        # Return the best individual found
        return np.max(self.func(self.population[:, :]))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# ```python
# ```python
# ```python