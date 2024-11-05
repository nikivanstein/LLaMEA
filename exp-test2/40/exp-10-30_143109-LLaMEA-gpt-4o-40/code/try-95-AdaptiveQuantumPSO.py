import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.island_size = 5
        self.num_islands = self.population_size // self.island_size
        self.global_best = None
        self.global_best_value = float('inf')
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, float('inf'))
        self.func_evals = 0
        self.c1 = 2.0  # Increased cognitive component for diverse search
        self.c2 = 2.0  # Increased social component
        self.F = 0.6  # Reduced mutation factor for stability
        self.CR = 0.9  # Increased crossover rate for more exploration
        self.elitism_rate = 0.1  # Decreased elitism to promote diversity
        self.q_factor = 0.5  # Quantum-inspired factor for position update

    def __call__(self, func):
        while self.func_evals < self.budget:
            for island_idx in range(self.num_islands):
                island_start = island_idx * self.island_size
                island_end = island_start + self.island_size
                island_positions = self.positions[island_start:island_end]
                island_best_idx = np.argmin(self.personal_best_values[island_start:island_end])
                island_best_position = island_positions[island_best_idx]

                for i in range(self.island_size):
                    idx = island_start + i
                    fitness = func(self.positions[idx])
                    self.func_evals += 1

                    if fitness < self.personal_best_values[idx]:
                        self.personal_best_values[idx] = fitness
                        self.personal_best_positions[idx] = self.positions[idx]

                    if fitness < self.global_best_value:
                        self.global_best_value = fitness
                        self.global_best = self.positions[idx]

                r1 = np.random.rand(self.island_size, self.dim)
                r2 = np.random.rand(self.island_size, self.dim)
                adaptive_learning_rate = 0.3 + np.random.rand(self.island_size, self.dim) * 0.7
                self.velocities[island_start:island_end] = (
                    adaptive_learning_rate * (self.velocities[island_start:island_end] +
                    self.c1 * r1 * (self.personal_best_positions[island_start:island_end] - island_positions) +
                    self.c2 * r2 * (self.global_best - island_positions))
                )
                self.positions[island_start:island_end] = island_positions + self.q_factor * np.sin(self.velocities[island_start:island_end])
                self.positions[island_start:island_end] = np.clip(self.positions[island_start:island_end], -5.0, 5.0)

            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.personal_best_values)[:elite_count]
            self.positions[:elite_count] = self.personal_best_positions[elite_indices]

        return self.global_best