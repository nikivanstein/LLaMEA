import numpy as np
import random

class ADMDF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.levy_flight_probability = 0.05
        self.self_adaptive_parameter_control = True
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')
        self.probability_map = np.random.uniform(0, 1, (self.population_size, self.dim))

    def levy_flight(self, x):
        sigma = 0.01
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, sigma)
        step_size = np.abs(u) / np.abs(v)
        return x + step_size * (x - np.random.uniform(-5.0, 5.0, self.dim))

    def self_adaptive_parameter_control(self):
        self.sigma = np.random.uniform(0.01, 0.1)
        self.F = np.random.uniform(0.1, 1.0)

    def crossover(self, x1, x2):
        if random.random() < self.crossover_probability:
            return (x1 + x2) / 2
        else:
            return x1

    def mutation(self, x):
        if random.random() < self.mutation_probability:
            return self.levy_flight(x)
        else:
            return x + np.random.uniform(-1.0, 1.0, self.dim)

    def probability_map_crossover(self, x1, x2):
        x_new = np.zeros(self.dim)
        for i in range(self.dim):
            if self.probability_map[i, np.argmin(x1)] > random.random():
                x_new[i] = x1[i]
            else:
                x_new[i] = x2[i]
        return x_new

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                x1 = population[i]
                x2 = np.random.uniform(-5.0, 5.0, self.dim)
                x_new = self.probability_map_crossover(x1, x2)
                x_new = self.mutation(x_new)
                if func(x_new) < func(x1):
                    population[i] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if self.self_adaptive_parameter_control:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)