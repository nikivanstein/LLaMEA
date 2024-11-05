import numpy as np

class EnhancedIslandPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.island_size = 4
        self.global_best = None
        self.global_best_value = float('inf')
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, float('inf'))
        self.func_evals = 0
        self.initial_temperature = 1.0
        self.temperature_decay = 0.95
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        while self.func_evals < self.budget:
            temperature = self.initial_temperature * (self.temperature_decay ** (self.func_evals / self.budget))
            for island_start in range(0, self.population_size, self.island_size):
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

                r1, r2 = np.random.rand(2)
                self.velocities[island_start:island_end] *= (1 - temperature)
                self.velocities[island_start:island_end] += (
                    self.c1 * r1 * (self.personal_best_positions[island_start:island_end] - island_positions) +
                    self.c2 * r2 * (self.global_best - island_positions)
                )
                self.positions[island_start:island_end] += self.velocities[island_start:island_end]
                self.positions[island_start:island_end] = np.clip(self.positions[island_start:island_end], -5.0, 5.0)

            if self.func_evals % (self.budget // 5) == 0:
                island_comm = np.random.randint(0, self.population_size, self.island_size)
                for i in island_comm:
                    if np.random.rand() < temperature:
                        self.positions[i] = self.global_best
                
        return self.global_best