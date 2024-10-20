import numpy as np
import random

class ADMDF_Chaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.levy_flight_probability = 0.05
        self.self_adaptive_parameter_control_probability = 0.05
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')
        self.chaos_map = np.array([[0, 0.3, 0.6], [0.3, 0, 0.7], [0.6, 0.7, 0]])

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
            r = np.random.uniform(-1.0, 1.0, self.dim)
            for i in range(self.dim):
                r[i] = self.chaos_map[i % 3, i % 3] * r[i] + (1 - self.chaos_map[i % 3, i % 3]) * x[i]
            return x + r

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                if random.random() < 0.21:  # Change the individual lines of the selected solution with probability 0.21
                    x_new = self.mutation(population[i])
                else:
                    x_new = self.crossover(population[i], self.levy_flight(population[i]))
                if func(x_new) < func(population[i]):
                    population[i] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if random.random() < self.self_adaptive_parameter_control_probability:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)