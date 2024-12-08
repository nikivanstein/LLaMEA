import numpy as np
import random
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = None
        self.best_individual = None
        self.best_value = -math.inf

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

    def initialize_population(self, func, num_individuals):
        self.population = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(num_individuals)]
        self.best_individual = min(self.population, key=func)
        self.best_value = func(self.best_individual)

    def mutate(self, individual):
        if random.random() < 0.1:
            index1, index2 = random.sample(range(len(individual)), 2)
            individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

    def iterated_permutation(self, func, num_iterations):
        for _ in range(num_iterations):
            new_population = []
            for individual in self.population:
                new_individual = individual.copy()
                self.population.append(new_individual)
                new_population.append(self.mutate(individual))
            self.population = new_population
            self.best_individual = min(self.population, key=func)
            self.best_value = func(self.best_individual)
        return self.population

    def cooling_schedule(self, initial_value, final_value, num_iterations):
        cooling_rate = initial_value / final_value
        value = initial_value
        for _ in range(num_iterations):
            value -= cooling_rate
            if value <= final_value:
                break
        return value

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 
# ```python
# BlackBoxOptimizer: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: 