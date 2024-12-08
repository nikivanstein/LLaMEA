import numpy as np
import random

class ADMDFS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.levy_flight_probability = 0.05
        self.self_adaptive_parameter_control_probability = 0.05
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')
        self.dynamic_mutation_probability = np.ones(self.population_size)

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

    def dynamic_mutation(self, x, i):
        if random.random() < self.dynamic_mutation_probability[i]:
            if random.random() < self.levy_flight_probability:
                return self.levy_flight(x)
            else:
                return x + np.random.uniform(-1.0, 1.0, self.dim)
        else:
            return x

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                x_new = self.dynamic_mutation(population[i], i)
                if func(x_new) < func(population[i]):
                    population[i] = x_new
                    self.dynamic_mutation_probability[i] = 0.8
                else:
                    self.dynamic_mutation_probability[i] = 0.2
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if random.random() < self.self_adaptive_parameter_control_probability:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)