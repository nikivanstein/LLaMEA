import numpy as np

class ProbabilisticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.best_position = np.zeros((self.population_size, self.dim))
        self.best_position_global = np.zeros(self.dim)
        self.best_fitness_global = np.inf
        self.perturbation_probability = 0.3

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.particles)
            for i in range(self.population_size):
                if np.random.rand() < self.perturbation_probability:
                    self.particles[i] += np.random.uniform(-1, 1, self.dim)
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                if fitness[i] < self.best_fitness_global:
                    self.best_position_global = self.particles[i]
                    self.best_fitness_global = fitness[i]
                if fitness[i] < self.best_fitness_global[i]:
                    self.best_position[i] = self.particles[i]
            self.velocity = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocity[i] = self.best_position[i] + r1 * (self.particles[i] - self.best_position[i])
                self.velocity[i] = np.clip(self.velocity[i], self.lower_bound, self.upper_bound)
            self.particles += self.velocity

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = ProbabilisticPSO(budget, dim)
best_position = optimizer(func)
print("Best position:", best_position)
print("Best fitness:", func(best_position))