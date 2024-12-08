import numpy as np
import random

class DAMEHL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.num_ensembles = 5
        self.learning_rate = 0.01
        self.crossover_probability = 0.5
        self.sigma = 0.1
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim, self.num_ensembles))
        self.fitness = np.inf * np.ones((self.population_size, self.num_ensembles))
        self.best_x = np.inf * np.ones((self.dim, self.num_ensembles))
        self.best_fitness = np.inf * np.ones(self.num_ensembles)

    def __call__(self, func):
        for i in range(self.budget):
            for j in range(self.num_ensembles):
                y = func(self.x[:, :, j])
                self.fitness[:, j] = y
                idx = np.argmin(y)
                self.best_x[:, j] = self.x[idx, :, j]
                self.best_fitness[j] = y[idx]
            for k in range(self.population_size):
                for l in range(self.num_ensembles):
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == k or r2 == k or r3 == k:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.x[r1, :, l] + self.F * (self.x[r2, :, l] - self.x[r3, :, l])
                    x_new = x_new + self.sigma * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[k, l]:
                        self.x[k, :, l] = x_new
                        self.fitness[k, l] = y_new
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[np.argmin(self.fitness[:, 0]), 0])
            if np.min(self.fitness[:, 0]) < self.best_fitness[0]:
                self.best_fitness[0] = np.min(self.fitness[:, 0])
                self.best_x[:, 0] = self.x[np.argmin(self.fitness[:, 0]), :, 0]
        return self.best_x[:, 0], self.best_fitness[0]

    def update_strategy(self, probability):
        if random.random() < probability:
            self.learning_rate = self.learning_rate * 0.9
            self.crossover_probability = self.crossover_probability * 0.9
            self.sigma = self.sigma * 0.9
        else:
            self.learning_rate = self.learning_rate * 1.1
            self.crossover_probability = self.crossover_probability * 1.1
            self.sigma = self.sigma * 1.1