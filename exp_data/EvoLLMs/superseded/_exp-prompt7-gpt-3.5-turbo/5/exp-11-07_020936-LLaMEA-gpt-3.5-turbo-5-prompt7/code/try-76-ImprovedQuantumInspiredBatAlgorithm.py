import numpy as np

class ImprovedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma

    def __call__(self, func):
        bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_solution, best_fitness = bats[0], func(bats[0])

        for _ in range(self.budget):
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequencies = np.clip(best_solution + self.alpha * (bats[i] - best_solution), -5.0, 5.0)
                    velocities[i] += frequencies * self.gamma
                else:
                    velocities[i] = np.random.uniform(-1, 1, self.dim) * np.linalg.norm(velocities[i]) or np.random.uniform(-1, 1, self.dim)

                new_solution = np.clip(bats[i] + velocities[i], -5.0, 5.0)
                new_fitness = func(new_solution)

                if np.random.rand() < self.loudness and new_fitness < best_fitness:
                    bats[i], best_solution, best_fitness = new_solution, new_solution, new_fitness

        return best_solution