import numpy as np

class ImprovedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma
        self.bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_solution, self.best_fitness = self.bats[0], func(self.bats[0])

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                self.pulse_rate = 0.9 * self.pulse_rate if np.random.rand() < 0.1 else 0.1 * self.pulse_rate
                if np.random.rand() > self.pulse_rate:
                    frequencies = np.clip(self.best_solution + self.alpha * (self.bats[i] - self.best_solution), -5.0, 5.0)
                    self.velocities[i] += frequencies * self.gamma
                else:
                    self.velocities[i] = np.random.uniform(-1, 1, self.dim) * np.linalg.norm(self.velocities[i]) if np.linalg.norm(self.velocities[i]) != 0 else np.random.uniform(-1, 1, self.dim)

                new_solution = np.clip(self.bats[i] + self.velocities[i], -5.0, 5.0)
                new_fitness = func(new_solution)

                if np.random.rand() < self.loudness and new_fitness < self.best_fitness:
                    self.bats[i], self.best_solution, self.best_fitness = new_solution, new_solution, new_fitness

        return self.best_solution