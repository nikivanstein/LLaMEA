import numpy as np

class EnhancedFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.positions)
            improved = fitness < self.personal_best_values
            self.personal_best_values[improved] = fitness[improved]
            self.personal_best_positions = np.where(improved[:, None], self.positions, self.personal_best_positions)

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.global_best_value:
                self.global_best_value = fitness[best_idx]
                self.global_best_position = self.positions[best_idx]

            r1, r2 = np.random.random((self.population_size, self.dim, 2))
            self.velocities = self.w * self.velocities + self.c1 * r1 * (self.personal_best_positions - self.positions) + self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
            self.positions = np.clip(self.positions + self.velocities, -5.0, 5.0)

        return self.global_best_value