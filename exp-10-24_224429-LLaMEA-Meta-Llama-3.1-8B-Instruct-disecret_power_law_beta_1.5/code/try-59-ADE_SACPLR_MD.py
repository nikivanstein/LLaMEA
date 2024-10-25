import numpy as np
import random

class ADE_SACPLR_MD:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.F = 0.5
        self.CR = 0.5
        self.sigma = 0.1
        self.learning_rate = 0.01
        self.crossover_probability = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.directions = np.random.normal(0, 1, (self.population_size, self.dim))

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            for j in range(self.population_size):
                if j!= idx:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == idx or r2 == idx or r3 == idx:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.x[r1] + self.F * (self.x[r2] - self.x[r3])
                    x_new = x_new + self.sigma * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
                        self.directions[j] = np.random.normal(0, 1, self.dim)
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            self.F = self.F + self.learning_rate * (self.fitness[idx] - self.best_fitness)
            self.F = max(0.1, min(2.0, self.F))
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness

# To update the selected solution, change the individual lines of the selected solution to refine its strategy with the probability 0.2391304347826087
# For example:
def update_solution(solution, probability):
    if random.random() < probability:
        solution.budget = 100  # change the budget
    if random.random() < probability:
        solution.F = 0.7  # change the F parameter
    if random.random() < probability:
        solution.CR = 0.3  # change the CR parameter
    if random.random() < probability:
        solution.sigma = 0.05  # change the sigma parameter
    return solution

# Usage:
algorithm = ADE_SACPLR_MD(budget=100, dim=10)
best_x, best_fitness = algorithm(func=your_function)
print("Best x:", best_x)
print("Best fitness:", best_fitness)