import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_particle_positions = np.copy(self.population)
        self.best_particle_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.alpha = 0.75  # Constriction factor
        self.beta = 1.0   # Quantum behavior influence
        self.phi = np.zeros((self.pop_size, self.dim))  # Local attractor

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_particle(self, idx):
        r1, r2 = np.random.rand(2)
        self.phi[idx] = self.alpha * self.best_particle_positions[idx] + (1 - self.alpha) * self.best_global_position
        for d in range(self.dim):
            self.population[idx, d] = self.phi[idx, d] + self.beta * (r1 - 0.5) * np.abs(self.phi[idx, d] - self.population[idx, d])
        self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.best_particle_fitness[i] = self.fitness[i]
            self.best_particle_positions[i] = np.copy(self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = np.copy(self.population[i])

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                self.update_particle(i)
                fitness = self.evaluate(func, self.population[i])
                if fitness < self.best_particle_fitness[i]:
                    self.best_particle_fitness[i] = fitness
                    self.best_particle_positions[i] = np.copy(self.population[i])
                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = np.copy(self.population[i])

        return self.best_global_position