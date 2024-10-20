import numpy as np
import random

class ADDE_SAFM_MPDS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5  # initial frequency
        self.CR = 0.5  # initial crossover rate
        self.pop_size = 50  # initial population size
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.best_solution = np.inf
        self.probability = 0.1  # probability of probabilistic exploration
        self.phases = 5  # number of optimization phases
        self.phase = 1  # current optimization phase
        self.phase_duration = self.budget // self.phases  # duration of each phase
        self.population_size_history = [self.pop_size] * self.phases  # population size history

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
            # dynamic population size
            if self.phase < self.phases:
                if self.phase == 1:
                    self.population_size_history[self.phase] = self.pop_size
                elif self.phase == 2:
                    self.population_size_history[self.phase] = self.pop_size // 2
                elif self.phase == 3:
                    self.population_size_history[self.phase] = self.pop_size // 3
                elif self.phase == 4:
                    self.population_size_history[self.phase] = self.pop_size // 4
                self.pop_size = self.population_size_history[self.phase]
                self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
            # phase transition
            if _ % self.phase_duration == 0 and self.phase < self.phases:
                self.phase += 1
        return self.best_solution

    def __call__(self, func):
        self.func = func
        return self.optimize(func)

# example usage
budget = 1000
dim = 10
optimizer = ADDE_SAFM_MPDS(budget, dim)
best_solution = optimizer(lambda x: np.sum(x**2))
print("Best solution:", best_solution)