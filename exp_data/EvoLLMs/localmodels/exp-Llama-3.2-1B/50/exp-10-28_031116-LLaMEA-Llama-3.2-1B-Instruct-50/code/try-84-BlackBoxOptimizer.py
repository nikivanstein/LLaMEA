import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = [[np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)] for _ in range(self.population_size)]
        self.fitness_scores = np.zeros((self.population_size, self.population_size, self.dim))
        self.variance = np.zeros((self.population_size, self.population_size, self.dim))

    def __call__(self, func):
        def evaluate_func(x):
            return func(x)

        def evaluate_fitness(x):
            return np.mean([evaluate_func(xi) for xi in x])

        for _ in range(self.budget):
            for i, x in enumerate(self.population):
                fitness = evaluate_fitness(x)
                self.fitness_scores[i, :, :] = fitness
                self.variance[i, :, :] = np.var([evaluate_fitness(xi) for xi in x])

        best_x = np.argmax(self.fitness_scores, axis=0)
        best_fitness = np.max(self.fitness_scores, axis=0)
        return best_x, best_fitness

    def mutate(self, x, mutation_rate):
        for _ in range(int(self.population_size * mutation_rate)):
            i, j = random.sample(range(self.population_size), 2)
            x[i], x[j] = x[j], x[i]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# black_box_optimizer = BlackBoxOptimizer(budget, dim)
# best_solution, best_fitness = black_box_optimizer(func)
# print("Best Solution:", best_solution)
# print("Best Fitness:", best_fitness)