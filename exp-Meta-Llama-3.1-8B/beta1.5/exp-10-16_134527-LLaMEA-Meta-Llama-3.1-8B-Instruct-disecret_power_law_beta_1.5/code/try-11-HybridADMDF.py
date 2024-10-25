import numpy as np
import random

class HybridADMDF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_probability = 0.1
        self.levy_flight_probability = 0.05
        self.self_adaptive_parameter_control = True
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')
        self.crossover_probability = 0.9
        self.mutation_step_size = 1.0

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
            if random.random() < self.levy_flight_probability:
                return self.levy_flight(x)
            else:
                return x + self.sigma * np.random.uniform(-1.0, 1.0, self.dim)
        else:
            return x

    def adapt_crossover_probability(self):
        if self.f_best < 1e-3:
            self.crossover_probability = 0.7
        elif self.f_best < 1e-2:
            self.crossover_probability = 0.8
        else:
            self.crossover_probability = 0.9

    def adapt_mutation_step_size(self):
        if self.f_best < 1e-3:
            self.mutation_step_size = 0.5
        elif self.f_best < 1e-2:
            self.mutation_step_size = 1.0
        else:
            self.mutation_step_size = 1.5

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                x_new = self.mutation(population[i])
                if func(x_new) < func(population[i]):
                    population[i] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            self.adapt_crossover_probability()
            self.adapt_mutation_step_size()
            if self.self_adaptive_parameter_control:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)