import numpy as np

class EnhancedQuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        bats = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))

        for _ in range(self.budget):
            frequencies = best_solution + self.alpha * (bats - best_solution)
            frequencies = np.clip(frequencies, -5.0, 5.0)
            update_mask = np.random.rand(self.population_size) > self.pulse_rate
            velocities[update_mask] += frequencies[update_mask] * self.gamma
            velocities[~update_mask] = np.random.uniform(-1, 1, (np.sum(~update_mask), self.dim)) * np.linalg.norm(velocities[~update_mask], axis=1)[:, None]
                
            new_solutions = bats + velocities
            new_solutions = np.clip(new_solutions, -5.0, 5.0)
            new_fitness = np.array([func(sol) for sol in new_solutions])

            update_mask = np.random.rand(self.population_size) < self.loudness
            improved_mask = new_fitness < best_fitness
            update_both_mask = update_mask & improved_mask
            bats[update_both_mask] = new_solutions[update_both_mask]
            best_solution = np.where(update_both_mask, new_solutions, best_solution)
            best_fitness = np.where(update_both_mask, new_fitness, best_fitness)

        return best_solution