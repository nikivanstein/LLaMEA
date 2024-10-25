import numpy as np
import random

class ADMDF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 1.0
        self.mutation_probability = 0.0
        self.levy_flight_probability = 0.05
        self.self_adaptive_parameter_control_probability = 0.0
        self.sigma = 0.1
        self.F = 0.5
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
        self.f_best = float('inf')

    def levy_flight(self, x):
        sigma = 0.01
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, sigma)
        step_size = np.abs(u) / np.abs(v)
        return x + step_size * (x - np.random.uniform(-5.0, 5.0, self.dim))

    def crossover(self, x1, x2):
        return (x1 + x2) / 2

    def update_selected_solution(self, x):
        x_new = self.levy_flight(x)
        x_new = self.crossover(x, x_new)
        return x_new

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                x_new = self.update_selected_solution(population[i])
                if func(x_new) < func(population[i]):
                    population[i] = x_new
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)