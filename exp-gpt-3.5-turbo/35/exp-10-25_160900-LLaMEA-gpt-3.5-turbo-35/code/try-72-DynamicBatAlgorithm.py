import numpy as np

class DynamicBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.best_solution = np.zeros(self.dim)
        self.best_fitness = float('inf')

    def __call__(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])

        while self.budget > 0:
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    new_solution = self.population[i] + self.alpha * (np.random.rand(self.dim) - 0.5)
                else:
                    new_solution = self.best_solution + self.gamma * np.random.normal(0, 1, self.dim)

                new_solution = np.clip(new_solution, -5.0, 5.0)
                new_fitness = func(new_solution)
                
                if (np.random.rand() < self.loudness) and (new_fitness <= self.fitness[i]):
                    self.population[i] = new_solution
                    self.fitness[i] = new_fitness

                    if new_fitness < self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = new_fitness

            self.loudness *= 0.99
            self.pulse_rate = 0.2 + 0.8 * (1 - np.exp(-0.05 * self.budget))

            self.budget -= self.population_size

        return self.best_solution