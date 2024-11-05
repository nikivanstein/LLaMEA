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
        self.c1 = 1.2  # Adaptive learning rate
        self.c2 = 1.7  # Adaptive learning rate
        self.F = 0.6  # Slightly tuned mutation factor
        self.CR = 0.9  # Increased crossover rate
        self.elitism_rate = 0.2  # Retaining rate for elite solutions

    def __call__(self, func):
        while self.func_evals < self.budget:
            for island_idx in range(self.num_islands):
                island_start = island_idx * self.island_size
                island_end = island_start + self.island_size
                
                neighborhood = np.random.choice(self.population_size, self.island_size, replace=False)
                neighborhood_best_idx = np.argmin(self.personal_best_values[neighborhood])

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
                velocity_constriction = 0.6 + np.random.rand(self.island_size, self.dim) * 0.3  # Self-adaptive velocity control
                self.velocities[island_start:island_end] = (
                    velocity_constriction * (self.velocities[island_start:island_end] +
                    self.c1 * r1 * (self.personal_best_positions[island_start:island_end] - self.positions[island_start:island_end]) +
                    self.c2 * r2 * (self.personal_best_positions[neighborhood[neighborhood_best_idx]] - self.positions[island_start:island_end]))
                )
                self.positions[island_start:island_end] += self.velocities[island_start:island_end]
                self.positions[island_start:island_end] = np.clip(self.positions[island_start:island_end], -5.0, 5.0)

                # Intermittent stochastic velocity reset
                if np.random.rand() < 0.05:
                    self.velocities[island_start:island_end] = np.random.uniform(-0.5, 0.5, (self.island_size, self.dim))

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