import numpy as np
import random

class AFMLF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5  # initial frequency
        self.CR = 0.5  # initial crossover rate
        self.pop_size = 50  # initial population size
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.best_solution = np.inf

    def evaluate(self, x):
        return self.func(x)

    def func(self, x):
        # BBOB test suite of 24 noiseless functions
        # Here we use the first function f1(x) = sum(x_i^2)
        return np.sum(x**2)

    def frequency_modulation(self, x):
        # frequency modulation function
        return self.F * np.sin(2 * np.pi * self.F * x)

    def adaptive_frequency_modulation(self):
        # adaptive frequency modulation
        self.F *= 1.1  # increase frequency
        if self.evaluate(self.population) > self.evaluate(self.population * self.frequency_modulation()):
            self.F *= 0.9  # decrease frequency
        return self.F

    def levy_flight(self, x):
        # Levy flight function
        sigma = 1
        step_size = np.random.normal(0, sigma)
        return x + step_size * np.random.uniform(-1, 1, self.dim)

    def crossover(self, x1, x2):
        # crossover operation
        r = np.random.rand(self.dim)
        return x1 + r * (x2 - x1)

    def selection(self, x1, x2):
        # selection operation
        if self.evaluate(x2) < self.evaluate(x1):
            return x2
        else:
            return x1

    def optimize(self, func):
        for _ in range(self.budget):
            # evaluate population
            fitness = [self.evaluate(x) for x in self.population]
            # get best solution
            self.best_solution = min(fitness)
            best_index = fitness.index(self.best_solution)
            # adaptive frequency modulation
            self.F = self.adaptive_frequency_modulation()
            # Levy flights
            for i in range(self.pop_size):
                # generate trial vector using Levy flight
                trial = self.population[i] + self.levy_flight(self.population[i])
                # evaluate trial vector
                trial_fitness = self.evaluate(trial)
                # selection
                self.population[i] = self.selection(self.population[i], trial)
                # update best solution
                if trial_fitness < self.evaluate(self.population[i]):
                    self.population[i] = trial
        return self.best_solution

    def __call__(self, func):
        self.func = func
        return self.optimize(func)

# example usage
budget = 1000
dim = 10
optimizer = AFMLF(budget, dim)
best_solution = optimizer(lambda x: np.sum(x**2))
print("Best solution:", best_solution)