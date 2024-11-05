import numpy as np

class EnhancedHybridPSO:
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
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.7
        self.CR = 0.8
        self.elitism_rate = 0.2

    def __call__(self, func):
        chaos_map = np.sin(np.linspace(0, np.pi, self.population_size))
        while self.func_evals < self.budget:
            for island_idx in range(self.num_islands):
                island_start = island_idx * self.island_size
                island_end = island_start + self.island_size
                island_positions = self.positions[island_start:island_end]
                island_velocities = self.velocities[island_start:island_end]
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

                adaptive_c1 = self.c1 + 0.1 * chaos_map[island_idx]
                adaptive_c2 = self.c2 + 0.1 * chaos_map[island_idx]
                velocity_constriction = 0.5 + np.random.rand(self.island_size, self.dim) * 0.2
                self.velocities[island_start:island_end] = (
                    velocity_constriction * (island_velocities +
                    adaptive_c1 * np.random.rand(self.island_size, self.dim) * (self.personal_best_positions[island_start:island_end] - island_positions) +
                    adaptive_c2 * np.random.rand(self.island_size, self.dim) * (self.global_best - island_positions))
                )
                self.positions[island_start:island_end] += self.velocities[island_start:island_end]
                self.positions[island_start:island_end] = np.clip(self.positions[island_start:island_end], -5.0, 5.0)

                if self.func_evals < self.budget * 0.5:
                    if np.random.rand() < 0.1:  # Occasional island restructuring
                        np.random.shuffle(self.positions)

            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.personal_best_values)[:elite_count]
            self.positions[:elite_count] = self.personal_best_positions[elite_indices]

        return self.global_best