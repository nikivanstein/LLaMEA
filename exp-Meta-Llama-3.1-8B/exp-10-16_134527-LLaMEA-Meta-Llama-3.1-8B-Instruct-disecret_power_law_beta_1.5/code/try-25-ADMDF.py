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
        self.self_adaptive_parameter_control_probability = 0.01
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')
        self.dynamic_population_size = np.linspace(50, 150, self.budget).astype(int)

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

    def multi_operator_crossover(self, x1, x2, x3):
        if random.random() < self.crossover_probability:
            return (x1 + x2) / 2
        elif random.random() < self.crossover_probability:
            return (x2 + x3) / 2
        else:
            return (x1 + x3) / 2

    def optimize(self, func):
        for i, pop_size in enumerate(self.dynamic_population_size):
            population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            for j in range(pop_size):
                x_new = self.mutation(population[j])
                if func(x_new) < func(population[j]):
                    population[j] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if random.random() < self.self_adaptive_parameter_control_probability:
                self.self_adaptive_parameter_control()
            if i < self.budget - 1:
                population2 = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
                for k in range(pop_size):
                    x_new = self.multi_operator_crossover(population[k], population2[k], self.x_best)
                    if func(x_new) < func(population[k]):
                        population[k] = x_new
                self.x_best = np.min(population, axis=0)
                self.f_best = func(self.x_best)
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)