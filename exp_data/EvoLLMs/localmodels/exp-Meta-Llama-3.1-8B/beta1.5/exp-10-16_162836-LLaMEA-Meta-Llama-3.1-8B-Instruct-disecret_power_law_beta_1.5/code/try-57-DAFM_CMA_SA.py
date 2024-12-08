import numpy as np
import random

class DAFM_CMA_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F = 0.5  # initial frequency
        self.CR = 0.5  # initial crossover rate
        self.pop_size = 50  # initial population size
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.best_solution = np.inf
        self.probability = 0.2  # increased probability of probabilistic exploration
        self.temperature = 100  # initial temperature for simulated annealing
        self.covariance_matrix = np.eye(self.dim)  # initial covariance matrix
        self.adaptive_step_size = 0.1  # initial adaptive step size for covariance matrix adaptation
        self.step_size_learning_rate = 0.001  # learning rate for adaptive step size
        self.history = np.zeros((self.budget, self.dim))  # history of best solutions
        self.dynamic_frequency_modulation = False  # flag for dynamic frequency modulation

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

    def dynamic_adaptive_frequency_modulation(self):
        # dynamic adaptive frequency modulation
        self.F = np.random.uniform(0.1, 10)  # randomly initialize frequency
        if self.dynamic_frequency_modulation:
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

    def simulated_annealing(self, x):
        # simulated annealing
        new_x = x + np.random.normal(0, 1, self.dim)
        new_x = np.clip(new_x, -5.0, 5.0)
        if self.evaluate(new_x) < self.evaluate(x):
            return new_x
        else:
            probability = np.exp(-(self.evaluate(new_x) - self.evaluate(x)) / self.temperature)
            if random.random() < probability:
                return new_x
            else:
                return x
        self.temperature *= 0.99  # decrease temperature with a higher rate

    def covariance_matrix_adaptation(self):
        # covariance matrix adaptation
        self.covariance_matrix = (1 - self.adaptive_step_size) * self.covariance_matrix + self.adaptive_step_size * np.eye(self.dim)
        return self.covariance_matrix

    def adaptive_step_size_learning(self):
        # adaptive step size learning
        self.adaptive_step_size += self.step_size_learning_rate * (self.evaluate(self.population[0]) - self.evaluate(self.population[-1]))
        return self.adaptive_step_size

    def history_learning(self):
        # history learning
        self.history = np.vstack((self.history, self.population[0]))

    def optimize(self, func):
        for i in range(self.budget):
            # evaluate population
            fitness = [self.evaluate(x) for x in self.population]
            # get best solution
            self.best_solution = min(fitness)
            best_index = fitness.index(self.best_solution)
            # dynamic adaptive frequency modulation
            if i % 10 == 0:
                self.dynamic_frequency_modulation = True
            else:
                self.dynamic_frequency_modulation = False
            self.F = self.dynamic_adaptive_frequency_modulation()
            # differential evolution
            for j in range(self.pop_size):
                # generate trial vector
                trial = self.crossover(self.population[j], self.population[random.randint(0, self.pop_size - 1)])
                # evaluate trial vector
                trial_fitness = self.evaluate(trial)
                # selection
                self.population[j] = self.selection(self.population[j], trial)
                # update best solution
                if trial_fitness < self.evaluate(self.population[j]):
                    self.population[j] = trial
            # simulated annealing
            for j in range(self.pop_size):
                self.population[j] = self.simulated_annealing(self.population[j])
            # covariance matrix adaptation
            self.covariance_matrix = self.covariance_matrix_adaptation()
            # adaptive step size learning
            self.adaptive_step_size = self.adaptive_step_size_learning()
            # history learning
            self.history_learning()
        return self.best_solution, self.history

    def __call__(self, func):
        self.func = func
        return self.optimize(func)

# example usage
budget = 1000
dim = 10
optimizer = DAFM_CMA_SA(budget, dim)
best_solution, history = optimizer(lambda x: np.sum(x**2))
print("Best solution:", best_solution)
print("History:", history)