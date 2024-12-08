import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.population_history = [[self.population]]  # to store the history of the population

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Generate a new population using Iterated Permutation and Cooling
            new_population = self.iterated_permutation_and_cooling(self.population, self.func, self.population_size, self.dim)
            # Evaluate the new population
            new_population_evals = self.evaluate_fitness(new_population, func)
            # Check if the new population is better
            if new_population_evals > self.func(np.random.uniform(-5.0, 5.0, self.dim)):
                # If the new population is better, replace the old population
                self.population = new_population
                self.population_history.append(self.population_history[-1])
                self.func_evals += new_population_evals
            else:
                # If the new population is not better, keep the old population
                self.population_history[-1].append(self.population)
                self.func_evals += new_population_evals
                # Cool down the cooling rate
                if len(self.population_history) > 100:
                    self.population_history.pop(0)
            # Select the fittest individual
            self.fittest_individual = self.select_fittest_individual()
        # Return the best individual found
        return self.fittest_individual

    def iterated_permutation_and_cooling(self, population, func, population_size, dim):
        new_population = population.copy()
        for _ in range(population_size):
            # Generate a random point in the search space
            point = np.random.uniform(-5.0, 5.0, dim)
            # Evaluate the function at the point
            value = func(point)
            # Check if the point is within the bounds
            if -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0:
                # If the point is within bounds, update the function value
                new_population = np.array([point])
                new_population_evals = func(new_population)
                # Check if the new point is better
                if new_population_evals > func(point):
                    # If the new point is better, replace the old point
                    new_population[0] = point
                    new_population_evals = func(new_population)
                    # Cool down the cooling rate
                    if len(self.population_history) > 100:
                        self.population_history.pop(0)
        return new_population

    def select_fittest_individual(self):
        # Select the fittest individual based on the fitness value
        # For simplicity, we use the fitness value directly
        return self.population[np.argmax(self.func(np.random.uniform(-5.0, 5.0, self.dim)))]

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# ```