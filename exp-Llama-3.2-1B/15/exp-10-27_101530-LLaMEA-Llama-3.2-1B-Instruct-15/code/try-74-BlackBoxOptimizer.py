import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        def mutate(individual):
            if random.random() < 0.15:
                index = random.randint(0, self.dim-1)
                new_value = func(individual[index])
                individual[index] = new_value
            return individual

        def evaluate_fitness(individual):
            func_value = func(individual)
            if self.func_evaluations < self.budget:
                self.func_evaluations += 1
                return func_value
            else:
                return self.search_space[0], self.search_space[1]

        best_individual = None
        best_fitness = float('-inf')
        for _ in range(1000):
            new_individual = evaluate_fitness(mutate(evaluate_fitness(individual)))
            if new_individual[0] > best_individual[0] or (new_individual[0] == best_individual[0] and new_individual[1] > best_individual[1]):
                best_individual = new_individual
                best_fitness = new_individual[0]
            individual = mutate(individual)

        return best_individual, best_fitness

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 