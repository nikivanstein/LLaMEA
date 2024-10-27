import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        def fitness(individual):
            return self.func(individual)

        def mutate(individual):
            new_individual = individual.copy()
            if random.random() < 0.05:
                new_individual[0] += random.uniform(-5.0, 5.0)
            return new_individual

        while True:
            fitness_values = [fitness(individual) for individual in self.search_space]
            fitness_values.sort(reverse=True)
            best_index = fitness_values.index(max(fitness_values))
            best_individual = self.search_space[best_index]

            # Refine strategy
            if random.random() < 0.05:
                new_individual = mutate(best_individual)
                self.search_space.append(new_individual)

            # Prune population
            if len(self.search_space) > self.budget:
                self.search_space.pop(0)

            return best_individual

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 