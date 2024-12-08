import numpy as np
import random

class ADDE_SAFM_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5  # initial frequency
        self.CR = 0.5  # initial crossover rate
        self.pop_size = 50  # initial population size
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.best_solution = np.inf
        self.probability = 0.1  # probability of probabilistic exploration
        self.min_pop_size = 20
        self.max_pop_size = 100

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
        if random.random() < self.probability:
            self.F *= 0.9  # decrease frequency with probability
        return self.F

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

    def dynamic_population_size_adjustment(self):
        # adjust population size based on convergence
        if self.budget > 200 and np.mean([self.evaluate(x) for x in self.population]) < self.best_solution:
            self.pop_size = min(self.pop_size + 10, self.max_pop_size)
        elif self.budget > 200 and np.mean([self.evaluate(x) for x in self.population]) >= self.best_solution:
            self.pop_size = max(self.pop_size - 10, self.min_pop_size)
        return self.pop_size

    def optimize(self, func):
        for _ in range(self.budget):
            # evaluate population
            fitness = [self.evaluate(x) for x in self.population]
            # get best solution
            self.best_solution = min(fitness)
            best_index = fitness.index(self.best_solution)
            # adaptive frequency modulation
            self.F = self.adaptive_frequency_modulation()
            # differential evolution
            self.pop_size = self.dynamic_population_size_adjustment()
            for i in range(self.pop_size):
                # generate trial vector
                trial = self.crossover(self.population[i], self.population[random.randint(0, self.pop_size - 1)])
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
optimizer = ADDE_SAFM_DPS(budget, dim)
best_solution = optimizer(lambda x: np.sum(x**2))
print("Best solution:", best_solution)