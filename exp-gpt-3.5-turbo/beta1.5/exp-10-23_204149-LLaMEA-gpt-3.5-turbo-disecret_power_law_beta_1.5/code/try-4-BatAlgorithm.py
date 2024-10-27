import numpy as np

class BatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.population = np.random.uniform(-5.0, 5.0, (population_size, dim))
        self.velocities = np.zeros((population_size, dim))
        self.frequencies = np.zeros(population_size)
        self.best_solution = np.copy(self.population[0])

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.population_size):
                if np.random.uniform() > self.pulse_rate:
                    self.frequencies[i] = np.random.uniform()
                self.velocities[i] += (self.population[i] - self.best_solution) * self.frequencies[i]
                new_solution = self.population[i] + self.velocities[i]
                if np.random.uniform() < self.loudness and func(new_solution) < func(self.population[i]):
                    self.population[i] = new_solution
                if func(self.population[i]) < func(self.best_solution):
                    self.best_solution = np.copy(self.population[i])
        return self.best_solution