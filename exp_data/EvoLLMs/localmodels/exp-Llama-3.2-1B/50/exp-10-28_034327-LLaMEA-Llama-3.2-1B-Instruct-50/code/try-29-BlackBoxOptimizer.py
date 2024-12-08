import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.best_individual = None
        self.best_fitness = -np.inf
        self.iterations = 0
        self.iteration_count = 0

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

    def iterated_permutation(self, func, budget):
        # Initialize the population with random individuals
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(budget)]

        while self.iterations < 1000:
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=self.func_evals, reverse=True)[:self.budget // 2]
            # Generate a new population by iterated permutation
            new_population = []
            for _ in range(self.budget):
                # Select a random individual from the fittest population
                individual = random.choice(fittest_individuals)
                # Generate a new point by iterated permutation
                new_point = []
                for _ in range(self.dim):
                    new_point.append(random.uniform(-5.0, 5.0))
                # Evaluate the function at the new point
                new_value = func(new_point)
                # Add the new individual to the new population
                new_population.append(new_point)
            # Replace the old population with the new population
            population = new_population
            # Update the best individual and fitness
            best_individual = max(population, key=self.func_evals)
            best_fitness = self.func_evals(best_individual)
            # If the best fitness is better than the current best, update the best individual and fitness
            if best_fitness > self.best_fitness:
                self.best_individual = best_individual
                self.best_fitness = best_fitness
            # Update the iteration count
            self.iterations += 1
            # If the iteration count is greater than or equal to 1000, break the loop
            if self.iterations >= 1000:
                break

        # Return the best individual and fitness
        return self.best_individual, self.best_fitness

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 