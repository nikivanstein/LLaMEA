import numpy as np
import random

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            dim = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(dim)
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_scores = []
            for func in self.population:
                func_score = func(func)
                fitness_scores.append((func_score, func))
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            selected_func = fitness_scores[0][1]
            selected_func_score = fitness_scores[0][0]
            self.population = [func for func in self.population if func!= selected_func]
            self.fitness_scores[self.population.index(selected_func)] = selected_func_score
            return selected_func