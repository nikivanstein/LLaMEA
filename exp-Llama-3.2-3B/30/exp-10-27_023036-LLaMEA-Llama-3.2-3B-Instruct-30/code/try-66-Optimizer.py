import numpy as np
import random

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_score = -np.inf
        self.pswarm = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            scores = np.zeros(self.population_size)
            for i in range(self.population_size):
                scores[i] = func(self.population[i])
            best_idx = np.argmin(scores)
            self.population[best_idx] = self.pswarm[best_idx]
            scores = np.zeros(self.population_size)
            for i in range(self.population_size):
                scores[i] = func(self.population[i])
            best_idx = np.argmin(scores)
            self.best_solution = self.population[best_idx]
            self.best_score = scores[best_idx]
            if self.best_score < self.best_score:
                self.pswarm[best_idx] = self.population[best_idx]
                self.pswarm = self.pswarm / np.sum(self.pswarm) * self.budget
                self.pswarm = self.pswarm + self.pswarm * np.random.normal(size=(self.population_size, self.dim))

            # Probabilistic mutation
            if random.random() < 0.3:
                self.population[best_idx] += np.random.uniform(-0.1, 0.1, self.dim)
                scores = np.zeros(self.population_size)
                for i in range(self.population_size):
                    scores[i] = func(self.population[i])
                best_idx = np.argmin(scores)
                self.best_solution = self.population[best_idx]
                self.best_score = scores[best_idx]

            # Probabilistic mutation
            if random.random() < 0.3:
                self.pswarm[best_idx] += np.random.uniform(-0.1, 0.1, self.dim)
                scores = np.zeros(self.population_size)
                for i in range(self.population_size):
                    scores[i] = func(self.pswarm[i])
                best_idx = np.argmin(scores)
                self.pswarm[best_idx] = self.population[best_idx]
                self.pswarm = self.pswarm / np.sum(self.pswarm) * self.budget
                self.pswarm = self.pswarm + self.pswarm * np.random.normal(size=(self.population_size, self.dim))

            # Probabilistic mutation
            if random.random() < 0.3:
                self.population[best_idx] += np.random.uniform(-0.1, 0.1, self.dim)
                scores = np.zeros(self.population_size)
                for i in range(self.population_size):
                    scores[i] = func(self.population[i])
                best_idx = np.argmin(scores)
                self.best_solution = self.population[best_idx]
                self.best_score = scores[best_idx]