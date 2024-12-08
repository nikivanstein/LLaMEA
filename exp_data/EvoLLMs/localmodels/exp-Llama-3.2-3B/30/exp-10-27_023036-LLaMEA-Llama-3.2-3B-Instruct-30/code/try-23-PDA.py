import numpy as np
import random

class PDA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()
        self.best_solution = None
        self.best_score = float('-inf')

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def evaluate(self, func):
        for solution in self.population:
            score = func(solution)
            self.population[np.argmin(self.population[:, 0])] = solution
            self.population[np.argmin(self.population[:, 0])] = solution
            self.population[np.argmin(self.population[:, 0])] = solution
        self.population = self.population[np.argsort(self.population[:, 0])]

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            if self.best_score < self.population[0, 0]:
                self.best_solution = self.population[0]
                self.best_score = self.population[0, 0]

        # Probability-driven adaptive strategy
        if self.best_solution is not None:
            probabilities = np.zeros(self.population_size)
            for i in range(self.population_size):
                probabilities[i] = np.exp(-((self.population[i, 0] - self.best_solution[0]) ** 2) / (2 * (self.population_size - 1)))

            selection_probabilities = np.random.choice(self.population_size, size=self.population_size, replace=True, p=probabilities)
            self.population = [self.population[i] for i in selection_probabilities]

            # Refine the best solution
            for i in range(self.population_size):
                if random.random() < 0.3:
                    self.population[i] += np.random.uniform(-0.1, 0.1, self.dim)
                    self.population[i] = np.clip(self.population[i], -5.0, 5.0)

        return self.population[0, 0]