import numpy as np

class ImprovedQuantumBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma

    def __call__(self, func):
        bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_solution, best_fitness = bats[0], func(bats[0])
        rand_vals = np.random.rand(self.budget)

        for i in range(self.budget):
            for j in range(self.population_size):
                if rand_vals[i] > self.pulse_rate:
                    frequencies = np.clip(best_solution + self.alpha * (bats[j] - best_solution), -5.0, 5.0)
                    velocities[j] += frequencies * self.gamma
                else:
                    velocities[j] = np.random.uniform(-1, 1, self.dim) * np.linalg.norm(velocities[j]) if np.linalg.norm(velocities[j]) else np.random.uniform(-1, 1, self.dim)
                new_solution = np.clip(bats[j] + velocities[j], -5.0, 5.0)
                new_fitness = func(new_solution)
                if rand_vals[i] < self.loudness and new_fitness < best_fitness:
                    bats[j], best_solution, best_fitness = new_solution, new_solution, new_fitness

        return best_solution