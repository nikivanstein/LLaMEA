import numpy as np

class HybridEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.island_size = 5  # Adjusted island size for better exploration
        self.num_islands = self.population_size // self.island_size
        self.global_best = None
        self.global_best_value = float('inf')
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, float('inf'))
        self.func_evals = 0
        self.constriction_factor = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.F = 0.5
        self.CR = 0.9
        self.elitism_rate = 0.1

    def __call__(self, func):
        memory_elite = np.copy(self.positions[0])  # Memory for best elite
        memory_elite_value = float('inf')
        
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

                r1 = np.random.rand(self.island_size, 1)  # Vectorized
                r2 = np.random.rand(self.island_size, 1)
                self.velocities[island_start:island_end] = (
                    self.constriction_factor * (island_velocities +
                    self.c1 * r1 * (self.personal_best_positions[island_start:island_end] - island_positions) +
                    self.c2 * r2 * (self.global_best - island_positions))
                )
                self.positions[island_start:island_end] += self.velocities[island_start:island_end]
                self.positions[island_start:island_end] = np.clip(self.positions[island_start:island_end], -5.0, 5.0)

                for i in range(self.island_size):
                    idx = island_start + i
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    self.F = 0.5 + 0.4 * np.random.rand()  # Enhanced adaptivity
                    mutant_vector = self.positions[a] + self.F * (self.positions[b] - self.positions[c])
                    trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[idx])
                    trial_vector = np.clip(trial_vector, -5.0, 5.0)
                    trial_fitness = func(trial_vector)
                    self.func_evals += 1
                    if trial_fitness < self.personal_best_values[idx]:
                        self.personal_best_values[idx] = trial_fitness
                        self.personal_best_positions[idx] = trial_vector

                if self.global_best_value < memory_elite_value:  # Update memory
                    memory_elite_value = self.global_best_value
                    memory_elite = np.copy(self.global_best)

            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.personal_best_values)[:elite_count]
            self.positions[:elite_count] = self.personal_best_positions[elite_indices]
            self.positions[elite_count - 1] = memory_elite  # Ensure memory elite is retained

        return self.global_best