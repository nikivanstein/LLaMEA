import numpy as np

class IslandPSO:
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

    def __call__(self, func):
        while self.func_evals < self.budget:
            for island_start in range(0, self.population_size, self.island_size):
                island_end = island_start + self.island_size
                island_positions = self.positions[island_start:island_end]
                island_velocities = self.velocities[island_start:island_end]
                island_best_position = island_positions[np.argmin(self.personal_best_values[island_start:island_end])]
                
                if self.func_evals > self.budget * 0.7:  # Introduce strategy change
                    restart_indices = np.random.rand(self.island_size) < 0.1
                    self.positions[island_start:island_end][restart_indices] = np.random.uniform(-5.0, 5.0, (restart_indices.sum(), self.dim))
                
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

                r1, r2, r3 = np.random.rand(3)
                self.velocities[island_start:island_end] = (
                    0.5 * island_velocities +  # Adjusted inertia weight
                    1.4 * r1 * (self.personal_best_positions[island_start:island_end] - island_positions) +
                    1.4 * r2 * (island_best_position - island_positions) +
                    0.3 * r3 * (self.global_best - island_positions)  # Increased influence of global best
                )
                self.positions[island_start:island_end] += self.velocities[island_start:island_end]
                crossover_mask = np.random.rand(self.island_size, self.dim) < 0.15  # Enhanced crossover rate
                self.positions[island_start:island_end] = np.where(crossover_mask, island_best_position, self.positions[island_start:island_end])
                self.positions[island_start:island_end] = np.clip(self.positions[island_start:island_end], -5.0, 5.0)
        
        return self.global_best