import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def novel_metaheuristic(self, func, budget, dim):
        # Initialize the population with random individuals
        population = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(100)]

        # Evolve the population over 100 generations
        for _ in range(100):
            # Select the fittest individuals
            fittest = sorted(population, key=func, reverse=True)[:int(budget/len(population))]

            # Create a new generation by linearly interpolating between the fittest individuals
            new_generation = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(len(fittest))]
            for i in range(len(fittest) - 1):
                new_generation[i] = fittest[i] + (fittest[i+1] - fittest[i]) * (np.random.uniform(0, 1) / 100)

            # Replace the old population with the new generation
            population = new_generation

            # If the budget is reached, return a default point and evaluation
            if len(population) == budget:
                return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Example usage:
optimizer = BlackBoxOptimizer(100, 5)
func = lambda x: x**2
best_individual, best_fitness = optimizer(novel_metaheuristic(func, 100, 5))
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")