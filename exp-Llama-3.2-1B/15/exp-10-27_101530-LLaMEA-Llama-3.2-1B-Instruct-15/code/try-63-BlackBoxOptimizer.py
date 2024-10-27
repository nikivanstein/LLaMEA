import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.new_individual = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.algorithms = {}

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        if random.random() < 0.1:  # 10% chance of mutation
            self.new_individual = individual[:]
            self.new_individual[1] = random.uniform(self.search_space[0], self.search_space[1])
        else:
            self.new_individual = individual

    def evaluate_fitness(self, individual, logger):
        fitness = func(individual)
        updated_fitness = min(self.best_fitness, fitness)
        if updated_fitness < self.best_fitness:
            self.best_fitness = updated_fitness
            self.best_individual = individual
            self.algorithms.clear()
        if logger:
            np.save(f"currentexp/aucs-{self.__class__.__name__}-{individual}.npy", fitness)
        return updated_fitness

    def update(self, algorithm_name, new_individual, logger):
        if algorithm_name in self.algorithms:
            self.algorithms[algorithm_name].mutate(new_individual)
        else:
            self.algorithms[algorithm_name] = BlackBoxOptimizer(
                self.budget, self.dim
            )
            self.algorithms[algorithm_name].mutate(new_individual)
        self.new_individual = new_individual
        self.best_individual = self.best_individual if self.best_individual else new_individual
        self.best_fitness = self.evaluate_fitness(self.best_individual, logger)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 