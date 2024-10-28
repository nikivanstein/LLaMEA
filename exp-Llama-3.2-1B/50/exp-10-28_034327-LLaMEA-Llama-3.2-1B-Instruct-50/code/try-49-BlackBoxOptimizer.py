import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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

    def __init_func(self, func):
        self.budget = 100
        self.dim = 2
        self.func = func
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

    def __init_init(self, func):
        self.budget = 100
        self.dim = 2
        self.func = func
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

    def __init_init_refined(self, func):
        self.budget = 100
        self.dim = 2
        self.func = func
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

    def __init_init_refined_refined(self, func):
        self.budget = 100
        self.dim = 2
        self.func = func
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

    def __iter(self):
        for _ in range(self.budget):
            while True:
                new_individual = self.evaluate_fitness(self.population)
                if new_individual == np.max(self.evaluate_fitness(self.population)):
                    break
                self.population.append(new_individual)
        return self.population

    def evaluate_fitness(self, individual):
        value = self.func(individual)
        if -5.0 <= individual[0] <= 5.0 and -5.0 <= individual[1] <= 5.0:
            return value
        return np.max(self.func(individual))

    def mutate(self, individual):
        index1, index2 = random.sample(range(self.dim), 2)
        individual[index1], individual[index2] = individual[index2], individual[index1]
        return individual

    def __str__(self):
        return f"Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm"

# Description: Novel Black Box Optimization using Iterated Permutation and Cooling Algorithm
# Code: