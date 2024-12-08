import numpy as np
import random

class ADMDF_MDLF:
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
        self.directions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.direction_probabilities = np.random.uniform(0.0, 1.0, self.population_size)

    def levy_flight(self, x, direction):
        sigma = 0.01
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, sigma)
        step_size = np.abs(u) / np.abs(v)
        return x + step_size * direction

    def self_adaptive_parameter_control(self):
        self.sigma = np.random.uniform(0.01, 0.1)
        self.F = np.random.uniform(0.1, 1.0)

    def crossover(self, x1, x2):
        if random.random() < self.crossover_probability:
            return (x1 + x2) / 2
        else:
            return x1

    def mutation(self, x, direction):
        if random.random() < self.mutation_probability:
            return self.levy_flight(x, direction)
        else:
            return x + np.random.uniform(-1.0, 1.0, self.dim)

    def update_direction_probabilities(self):
        self.direction_probabilities = np.exp(self.directions / np.sum(self.directions, axis=1, keepdims=True))

    def select_direction(self):
        return np.random.choice(self.population_size, p=self.direction_probabilities)

    def optimize(self, func):
        for _ in range(self.budget):
            population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            directions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            for i in range(self.population_size):
                direction_probability = np.random.uniform(0.0, 1.0)
                if direction_probability < 0.5:
                    direction_index = self.select_direction()
                    direction = directions[direction_index]
                else:
                    direction = np.random.uniform(-5.0, 5.0, self.dim)
                x_new = self.mutation(population[i], direction)
                if func(x_new) < func(population[i]):
                    population[i] = x_new
                    directions[i] = direction
            self.update_direction_probabilities()
            self.x_best = np.min(population, axis=0)
            self.f_best = func(self.x_best)
            if self.self_adaptive_parameter_control:
                self.self_adaptive_parameter_control()
        return self.x_best, self.f_best

    def __call__(self, func):
        return self.optimize(func)