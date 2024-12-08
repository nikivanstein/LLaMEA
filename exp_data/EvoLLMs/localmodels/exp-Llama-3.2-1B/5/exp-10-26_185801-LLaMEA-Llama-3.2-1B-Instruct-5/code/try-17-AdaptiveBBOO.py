import numpy as np
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = np.random.choice(self.search_space, self.population_size, replace=False)
        self.fitness_scores = np.zeros(self.population_size)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        if random.random() < 0.05:
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.05:
            index1, index2 = random.sample(range(self.search_space.shape[0]), 2)
            parent1[index1], parent2[index2] = parent2[index2], parent1[index1]
        return np.concatenate((parent1, parent2))

    def selection(self):
        self.population = np.array([self.population[i] for i in np.random.choice(self.population_size, size=self.population_size, replace=False)])
        self.fitness_scores = np.array([self.__call__(func) for func in self.population])
        self.population = np.array([self.population[i] for i in np.argsort(self.fitness_scores)[::-1]])