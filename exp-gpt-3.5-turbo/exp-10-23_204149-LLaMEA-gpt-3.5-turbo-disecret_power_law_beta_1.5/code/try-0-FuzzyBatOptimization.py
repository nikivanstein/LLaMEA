import numpy as np

class FuzzyBatOptimization:
    def __init__(self, budget, dim, population_size=10, alpha=0.9, gamma=0.9, fmin=0, fmax=1, loudness=1.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.gamma = gamma
        self.fmin = fmin
        self.fmax = fmax
        self.loudness = loudness
        self.population = np.random.uniform(-5.0, 5.0, (population_size, dim))
        self.velocities = np.zeros((population_size, dim))

    def __call__(self, func):
        for t in range(self.budget):
            frequencies = self.fmin + (self.fmax - self.fmin) * np.random.rand(self.population_size)
            for i in range(self.population_size):
                rand = np.random.uniform(-1, 1, self.dim)
                self.velocities[i] += (self.population[i] - self.population[np.random.randint(self.population_size)]) * frequencies[i]
                if np.random.rand() > self.alpha:
                    self.velocities[i] = self.velocities[i] * np.exp(self.gamma * t)
                new_solution = self.population[i] + self.velocities[i]
                if func(new_solution) < func(self.population[i]) and np.random.rand() < self.loudness:
                    self.population[i] = new_solution
        return self.population[np.argmin([func(ind) for ind in self.population])]