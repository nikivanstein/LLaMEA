import numpy as np

class EnhancedQuantumBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma

    def __call__(self, func):
        bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_solution, best_fitness = bats[0].copy(), func(bats[0])
        rand_vals = np.random.rand(self.budget)
        alpha_bats = self.alpha * bats

        for i in range(self.budget):
            rand_compare = np.random.rand(self.population_size)
            frequencies = np.clip(best_solution + alpha_bats - best_solution, -5.0, 5.0)
            velocities += np.where(rand_vals[i] > self.pulse_rate, frequencies, np.where(np.linalg.norm(velocities, axis=1).reshape(-1, 1), np.random.uniform(-1, 1, (self.population_size, self.dim)), np.random.uniform(-1, 1, (self.population_size, self.dim))))
            new_solutions = np.clip(bats + velocities, -5.0, 5.0)
            new_fitness = np.array([func(new_solutions[k]) for k in range(self.population_size)])
            update_indices = np.where(np.logical_and(rand_vals[i] < self.loudness, new_fitness < best_fitness))
            bats[update_indices], best_solution[update_indices], best_fitness[update_indices] = new_solutions[update_indices], new_solutions[update_indices], new_fitness[update_indices]

        return best_solution