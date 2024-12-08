import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, self.dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                self.func_evals += 1
                return value
        # If the budget is exceeded, return the best point found so far
        return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

    def __next_generation(self, population):
        # Select the fittest individuals
        fittest_individuals = population[np.argsort(self.func_evals)][::-1][:self.population_size // 2]
        # Select the next generation size
        next_generation_size = min(self.population_size, self.budget - self.func_evals)
        # Select the next generation individuals
        next_generation = np.random.choice(fittest_individuals, next_generation_size, replace=False)
        return next_generation

    def __next_point(self, population):
        # Select the next generation individuals
        next_generation = self.__next_generation(population)
        # Replace the old population with the new one
        population = np.concatenate((population, next_generation), axis=0)

    def optimize(self, func):
        # Initialize the population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        # Run the optimization algorithm
        while True:
            self.__next_point(population)
            # Evaluate the function at the new population
            new_value = func(population)
            # Update the best value
            if new_value > np.max(func(np.random.uniform(-5.0, 5.0, self.dim))):
                return new_value
            # If the budget is exceeded, return the best point found so far
            return np.max(func(np.random.uniform(-5.0, 5.0, self.dim)))

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 