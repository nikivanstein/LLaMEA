import numpy as np
import random

class HarmonySearchGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.harmONY_rate = 0.01
        self.HS = 0.5
        self.GWO = 0.5
        self.p = 0.021739130434782608
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf
        self.selected_solution = None

    def __call__(self, func):
        for i in range(self.budget):
            y = func(self.x)
            self.fitness = y
            idx = np.argmin(y)
            self.best_x = self.x[idx]
            self.best_fitness = y[idx]
            self.selected_solution = np.random.choice(self.population_size, p=self.fitness / np.sum(self.fitness))
            self.selected_solution = int(self.selected_solution)
            for j in range(self.population_size):
                if j!= self.selected_solution:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)
                    while r1 == self.selected_solution or r2 == self.selected_solution or r3 == self.selected_solution:
                        r1, r2, r3 = random.sample(range(self.population_size), 3)
                    x_new = self.x[r1] + self.HS * (self.x[r2] - self.x[r3])
                    x_new = x_new + self.GWO * np.random.normal(0, 1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
            if np.random.rand() < self.p:
                self.HS = self.HS + 0.01 * (self.HS - np.mean(self.fitness))
                self.GWO = self.GWO + 0.01 * (self.GWO - np.mean(self.fitness))
        return self.best_x, self.best_fitness