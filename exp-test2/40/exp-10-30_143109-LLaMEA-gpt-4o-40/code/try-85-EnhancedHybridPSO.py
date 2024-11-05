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
        # Chaotic initialization for positions
        self.positions = -5.0 + 10.0 * np.sin(np.linspace(0, np.pi, self.population_size * self.dim)).reshape(self.population_size, self.dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, float('inf'))
        self.func_evals = 0
        # Temporal adaptive parameters
        self.initial_c1 = 1.5
        self.final_c1 = 0.5
        self.initial_c2 = 1.5
        self.final_c2 = 2.5
        self.F = 0.7
        self.CR = 0.9
        self.elitism_rate = 0.3  # Increased elitism rate

    def __call__(self, func):
        while self.func_evals < self.budget:
            for island_idx in range(self.num_islands):
                island_start = island_idx * self.island_size
                island_end = island_start + self.island_size
                island_positions = self.positions[island_start:island_end]
                island_velocities = self.velocities[island_start:island_end]
                island_best_idx = np.argmin(self.personal_best_values[island_start:island_end])
                island_best_position = island_positions[island_best_idx]
                
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

                # Adaptation of c1 and c2 over time
                c1 = self.initial_c1 - (self.initial_c1 - self.final_c1) * (self.func_evals / self.budget)
                c2 = self.initial_c2 + (self.final_c2 - self.initial_c2) * (self.func_evals / self.budget)

                r1 = np.random.rand(self.island_size, self.dim)
                r2 = np.random.rand(self.island_size, self.dim)
                self.velocities[island_start:island_end] = (
                    0.5 * (island_velocities +
                    c1 * r1 * (self.personal_best_positions[island_start:island_end] - island_positions) +
                    c2 * r2 * (self.global_best - island_positions))
                )
                self.positions[island_start:island_end] += self.velocities[island_start:island_end]
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