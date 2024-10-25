import numpy as np
import random

class Hybrid_ADE_SACPLR_BFOA:
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
        self.chance_to_change = 0.021739130434782608
        self.n_steps = 0

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
            if np.random.rand() < self.chance_to_change:
                self.n_steps += 1
                if self.n_steps % 5 == 0:
                    self.F = self.F + np.random.uniform(-0.1, 0.1)
                    self.CR = self.CR + np.random.uniform(-0.1, 0.1)
                    self.sigma = self.sigma + np.random.uniform(-0.1, 0.1)
                if self.n_steps % 10 == 0:
                    self.learning_rate = self.learning_rate + np.random.uniform(-0.01, 0.01)
                    self.crossover_probability = self.crossover_probability + np.random.uniform(-0.1, 0.1)
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
        return self.best_x, self.best_fitness

class BacterialForagingOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.swarm_size = 50
        self.c1 = 0.1
        self.c2 = 0.1
        self.c3 = 0.1
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.best_x = np.inf * np.ones(self.dim)
        self.best_fitness = np.inf

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
                    x_new = self.x[r1] + self.c1 * np.random.normal(0, 1, self.dim) + self.c2 * (self.x[r2] - self.x[r3]) + self.c3 * np.random.normal(0, 1, self.dim)
                    x_new = x_new + np.random.normal(0, 0.1, self.dim)
                    x_new = np.clip(x_new, self.lower_bound, self.upper_bound)
                    y_new = func(x_new)
                    if y_new < self.fitness[j]:
                        self.x[j] = x_new
                        self.fitness[j] = y_new
        return self.best_x, self.best_fitness

class Hybrid_ADE_SACPLR_BFOA:
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
        self.chance_to_change = 0.021739130434782608
        self.n_steps = 0
        self.bfoa = BacterialForagingOptimization(self.budget, self.dim)

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
            if np.random.rand() < self.chance_to_change:
                self.n_steps += 1
                if self.n_steps % 5 == 0:
                    self.F = self.F + np.random.uniform(-0.1, 0.1)
                    self.CR = self.CR + np.random.uniform(-0.1, 0.1)
                    self.sigma = self.sigma + np.random.uniform(-0.1, 0.1)
                if self.n_steps % 10 == 0:
                    self.learning_rate = self.learning_rate + np.random.uniform(-0.01, 0.01)
                    self.crossover_probability = self.crossover_probability + np.random.uniform(-0.1, 0.1)
            self.CR = self.CR + self.learning_rate * (self.crossover_probability - self.CR)
            self.crossover_probability = max(0.1, min(1.0, self.CR))
            self.sigma = self.sigma + self.learning_rate * (self.sigma - self.fitness[idx])
            if self.fitness[idx] < self.best_fitness:
                self.best_fitness = self.fitness[idx]
                self.best_x = self.x[idx]
            self.bfoa.x = self.x
            self.bfoa.fitness = self.fitness
            best_x_bfoa, best_fitness_bfoa = self.bfoa(func)
            if best_fitness_bfoa < self.best_fitness:
                self.best_x = best_x_bfoa
                self.best_fitness = best_fitness_bfoa
        return self.best_x, self.best_fitness