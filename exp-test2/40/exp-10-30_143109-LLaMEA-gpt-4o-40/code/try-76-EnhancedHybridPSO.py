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
        self.chaos_parameter = 0.5  # Added chaos parameter

    def chaotic_map(self, x):
        return 0.9 * x + 0.1 * x * (1 - x)  # Logistic map for chaotic behavior

    def __call__(self, func):
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

                r1 = np.random.rand(self.island_size, self.dim)
                r2 = np.random.rand(self.island_size, self.dim)
                velocity_constriction = 0.5 + np.random.rand(self.island_size, self.dim) * 0.2  
                self.velocities[island_start:island_end] = (
                    velocity_constriction * (island_velocities +
                    self.c1 * r1 * (self.personal_best_positions[island_start:island_end] - island_positions) +
                    self.c2 * r2 * (self.global_best - island_positions))  # Use global best for stronger pull
                )

                self.positions[island_start:island_end] += self.chaotic_map(self.velocities[island_start:island_end])  # Apply chaotic map
                self.positions[island_start:island_end] = np.clip(self.positions[island_start:island_end], -5.0, 5.0)

                for i in range(self.island_size):
                    idx = island_start + i
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    trial_vector = np.where(np.random.rand(self.dim) < self.CR, 
                                            self.positions[a] + self.F * (self.positions[b] - self.positions[c]), 
                                            self.positions[idx])
                    trial_vector = np.clip(trial_vector, -5.0, 5.0)
                    trial_fitness = func(trial_vector)
                    self.func_evals += 1
                    if trial_fitness < self.personal_best_values[idx]:
                        self.personal_best_values[idx] = trial_fitness
                        self.personal_best_positions[idx] = trial_vector

            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.personal_best_values)[:elite_count]
            self.positions[:elite_count] = self.personal_best_positions[elite_indices]

        return self.global_best