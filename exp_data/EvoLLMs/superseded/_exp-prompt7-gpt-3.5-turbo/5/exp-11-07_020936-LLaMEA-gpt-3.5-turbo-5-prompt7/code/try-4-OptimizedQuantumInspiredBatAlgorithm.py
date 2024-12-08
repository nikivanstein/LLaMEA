import numpy as np

class OptimizedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget, self.dim, self.population_size, self.loudness, self.pulse_rate, self.alpha, self.gamma = budget, dim, population_size, loudness, pulse_rate, alpha, gamma

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))

        for _ in range(self.budget):
            update_indices = np.random.rand(self.population_size) > self.pulse_rate
            frequencies = best_solution + self.alpha * (bats - best_solution)
            frequencies = np.clip(frequencies, -5.0, 5.0)
            velocities[update_indices] += frequencies[update_indices] * self.gamma
            velocities[~update_indices] = np.random.uniform(-1, 1, (sum(~update_indices), self.dim)) * np.linalg.norm(velocities[~update_indices], axis=1)[:, None]
            
            new_solutions = bats + velocities
            new_solutions = np.clip(new_solutions, -5.0, 5.0)
            new_fitness = np.array([func(new_sol) for new_sol in new_solutions])

            loud_update_indices = np.random.rand(self.population_size) < self.loudness
            better_indices = new_fitness < best_fitness
            update_indices = loud_update_indices & better_indices

            bats[update_indices] = new_solutions[update_indices]
            best_solution = new_solutions[np.argmin(new_fitness)]
            best_fitness = min(new_fitness[update_indices])

        return best_solution